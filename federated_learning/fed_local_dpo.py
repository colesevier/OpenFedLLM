import torch
import copy
try:
    from trl import DPOTrainer, DPOConfig
except Exception:
    from trl import DPOTrainer
    DPOConfig = None
from transformers import Trainer as HFTrainer
from .fed_local_sft import SCAFFOLD_Callback

def _maybe_make_dpo_config(training_args, seq_length=512, beta=0.1):
    if DPOConfig is None:
        return training_args
    if isinstance(training_args, DPOConfig):
        return training_args
    # Build a minimal DPOConfig from TrainingArguments
    return DPOConfig(
        output_dir=training_args.output_dir,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        learning_rate=training_args.learning_rate,
        logging_steps=training_args.logging_steps,
        num_train_epochs=training_args.num_train_epochs,
        max_steps=training_args.max_steps,
        report_to=training_args.report_to,
        save_steps=training_args.save_steps,
        save_total_limit=training_args.save_total_limit,
        push_to_hub=training_args.push_to_hub,
        hub_model_id=training_args.hub_model_id,
        gradient_checkpointing=training_args.gradient_checkpointing,
        lr_scheduler_type=training_args.lr_scheduler_type,
        remove_unused_columns=False,
        generate_during_eval=False,
        max_length=seq_length,
        max_prompt_length=min(128, seq_length),
        bf16=False,
        fp16=False,
        tf32=None,
        beta=beta,
    )

class CompatDPOTrainer(DPOTrainer):
    # Make get_batch_samples signature compatible with HF Trainer 4.55+
    def get_batch_samples(self, epoch_iterator, num_batches, device):
        return HFTrainer.get_batch_samples(self, epoch_iterator, num_batches, device)

def get_fed_local_dpo_trainer(script_args, fed_args, model, model_ref, tokenizer, training_args, local_dataset, global_dict, local_auxiliary, global_auxiliary):
    
    if fed_args.fed_alg == 'fedprox':
        trainer = DPOTrainerFedProx(
                            model=model,
                            ref_model=model_ref,
                            args=_maybe_make_dpo_config(training_args, seq_length=script_args.seq_length, beta=script_args.dpo_beta),
                            train_dataset=local_dataset,
                            processing_class=tokenizer,
                            global_state=global_dict,
                            prox_mu=fed_args.prox_mu,
                            )
    elif fed_args.fed_alg == 'scaffold':
        trainer = DPOTrainerSCAFFOLD(
                            model=model,
                            ref_model=model_ref,
                            args=_maybe_make_dpo_config(training_args, seq_length=script_args.seq_length, beta=script_args.dpo_beta),
                            train_dataset=local_dataset,
                            processing_class=tokenizer,
                            global_state=global_dict,
                            local_auxiliary=local_auxiliary,
                            global_auxiliary=global_auxiliary,
                            )
        trainer.add_callback(SCAFFOLD_Callback(trainer.correction, model))
    elif (fed_args.fed_alg in ['fedavg', 'fedavgm', 'fedadgrad', 'fedyogi', 'fedadam']) or (fed_args.fed_alg).startswith('local'):
        if not hasattr(training_args, 'model_init_kwargs'):
            setattr(training_args, 'model_init_kwargs', None)
        if not hasattr(training_args, 'ref_model_init_kwargs'):
            setattr(training_args, 'ref_model_init_kwargs', None)
        trainer = CompatDPOTrainer(
                            model=model,
                            ref_model=model_ref,
                            args=_maybe_make_dpo_config(training_args, seq_length=script_args.seq_length, beta=script_args.dpo_beta),
                            train_dataset=local_dataset,
                            processing_class=tokenizer,
                            )
    else:
        raise ValueError(f'Unsupported `fed_alg`: {fed_args.fed_alg}')
    return trainer

class DPOTrainerFedProx(CompatDPOTrainer):
    def __init__(self, global_state, prox_mu, **kwargs):
        super(DPOTrainerFedProx, self).__init__(**kwargs)
        self.global_state = global_state
        self.mu = prox_mu
    
    def compute_loss(self, model, inputs, return_outputs=False):

        return_values = super(DPOTrainerFedProx, self).compute_loss(model, inputs, return_outputs=return_outputs)

        if return_outputs:
            loss, outputs = return_values
        else:
            loss = return_values

        # Apply FedProx Loss
        for name, param in model.named_parameters():
            name = name.replace(".default", "")     # TODO: May need changes. to accord with peft
            # only trainable parameters
            if not param.requires_grad:
                continue
            else:
                loss += self.mu / 2 * torch.norm(param - self.global_state[name]) ** 2

        return (loss, outputs) if return_outputs else loss
    
class DPOTrainerSCAFFOLD(CompatDPOTrainer):
    def __init__(self, global_state, local_auxiliary, global_auxiliary, **kwargs):
        super(DPOTrainerSCAFFOLD, self).__init__(**kwargs)
        self.global_state = global_state
        self.local_auxiliary = local_auxiliary
        self.global_auxiliary = global_auxiliary
        self.correction = copy.deepcopy(local_auxiliary)

        for name in self.correction.keys():
            self.correction[name] = self.global_auxiliary[name] - self.local_auxiliary[name]
    
    def get_auxiliary_param(self):
        auxiliary_new_para = copy.deepcopy(self.local_auxiliary)
        auxiliary_delta_para = copy.deepcopy(self.local_auxiliary)
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                else:
                    name = name.replace(".default", "")
                    auxiliary_new_para[name] = (self.global_state[name] - param) / (self.args.max_steps * self.args.learning_rate) - self.correction[name]
                    auxiliary_delta_para[name] = auxiliary_new_para[name] - self.local_auxiliary[name]

        return auxiliary_new_para, auxiliary_delta_para