import torch


class MyClass:
    def __init__(self):
        pass

    def method1(self):
        ckpt_path="/HDD_data_storage_2u_1/jinruiyang/shared_space/code/TinyMIM/TinyMIM-PT-B.pth"
        checkpoint = torch.load(ckpt_path, map_location='cpu')['model']  # Use 'cuda' for GPU

        for name,value in checkpoint.items():
            print(name,"\t", value.shape)

    def method2(self):
    #    ckpt_path="/HDD_data_storage_2u_1/jinruiyang/shared_space/code/SCLIP/clip/converted_checkpoints/crate_alpha_B16.pth"
       ckpt_path = '/HDD_data_storage_2u_1/jinruiyang/shared_space/code/SCLIP/clip/converted_checkpoints/B32_ablation_in21k_mlp_nodecouple_x1_mixup_open_warm10_4096_lr5e5_91e_norangaug_no_label_sm_v3_128_checkpoint.pth' # vanilla crate b32 on 21k
       checkpoint = torch.load(ckpt_path, map_location='cpu') # Use 'cuda' for GPU

       for name,value in checkpoint.items():
           print(name,"\t", value.shape)

    def convert_ckpt(self):
        ckpt_path="/HDD_data_storage_2u_1/jinruiyang/shared_space/code/SCLIP/clip/converted_checkpoints/crate_alpha_B16.pth"
        state_dict = torch.load(ckpt_path, map_location='cpu') # Use 'cuda' for GPU
        rename_state = {}
        model_prefix = 'blocks'
        ckpt_prefix = 'transformer.layers'
        for k,v in state_dict.items():
            if 'conv1' in k:
                if "conv1.weight" in k:
                    rename_state['patch_embed.proj.weight'] = v
                elif "conv1.bias" in k:
                    rename_state['patch_embed.proj.bias'] = v
                else:
                    assert False
            elif 'pos_embedding' in k:
                rename_state['pos_embed'] = v
            elif 'transformer.layers' in k:
                index = k.split('.')[2]
                rename_state[f'{model_prefix}.{index}.norm1.weight'] = state_dict[f'{ckpt_prefix}.{index}.0.norm.weight']
                rename_state[f'{model_prefix}.{index}.norm1.bias'] = state_dict[f'{ckpt_prefix}.{index}.0.norm.bias']

                rename_state[f'{model_prefix}.{index}.attn.qkv.weight'] = state_dict[f'{ckpt_prefix}.{index}.0.fn.qkv.weight']
                rename_state[f'{model_prefix}.{index}.attn.to_out.0.weight'] = state_dict[f'{ckpt_prefix}.{index}.0.fn.to_out.0.weight']    
                
                rename_state[f'{model_prefix}.{index}.norm2.weight'] = state_dict[f'{ckpt_prefix}.{index}.1.norm.weight']
                rename_state[f'{model_prefix}.{index}.norm2.bias'] = state_dict[f'{ckpt_prefix}.{index}.1.norm.bias']

                
                rename_state[f'{model_prefix}.{index}.mlp.D'] = state_dict[f'{ckpt_prefix}.{index}.1.fn.D']
                rename_state[f'{model_prefix}.{index}.mlp.D1'] = state_dict[f'{ckpt_prefix}.{index}.1.fn.D1']
            else:
                rename_state[k] = v

        for k,v in rename_state.items():
            print(k,v.shape)


# Creating an instance of MyClass
my_object = MyClass()
my_object.method2()
# my_object.convert_ckpt()