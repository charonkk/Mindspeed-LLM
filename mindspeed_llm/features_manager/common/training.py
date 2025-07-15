from mindspeed.features_manager.feature import MindSpeedFeature


class TrainingDefaultFeature(MindSpeedFeature):
    def __init__(self):
        super(TrainingDefaultFeature, self).__init__('train-default', optimization_level=0)

    def register_patches(self, patch_manager, args):
        from mindspeed_llm.training import train
        from mindspeed_llm.training.checkpointing import load_checkpoint_wrapper
        from mindspeed_llm.legacy.data import build_pretraining_data_loader
        from mindspeed_llm.tasks.posttrain.lora.utils import is_enable_qlora
        
        if is_enable_qlora(args):
            from mindspeed_llm.tasks.posttrain.lora.qlora import get_model
            patch_manager.register_patch('megatron.training.training.get_model', get_model)
        else:
            from mindspeed_llm.training import get_model_wrapper
            patch_manager.register_patch('megatron.training.training.get_model', get_model_wrapper)
        
        patch_manager.register_patch('megatron.training.training.build_pretraining_data_loader',
                                    build_pretraining_data_loader)
        patch_manager.register_patch('megatron.training.training.train', train)
        patch_manager.register_patch('megatron.training.training.load_checkpoint', load_checkpoint_wrapper)