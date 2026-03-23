"""
RosettaModel 推理示例。

这个脚本主要演示 4 件事：
1. 如何加载一个由 SLM、LLM 和 projector 组成的 RosettaModel。
2. 如何把聊天格式的 prompt 处理成模型可接受的输入张量。
3. 如何构造 `kv_cache_index`，让 Rosetta 在生成时知道哪些位置使用哪类缓存。
4. 如何对比 Rosetta、底层 SLM、底层 LLM 三种生成结果，便于快速做效果排查。

文件里还保留了一些被注释掉的实验代码，方便后续切换到更细粒度的
token 对齐或手动 prefill 调试流程。
"""

import torch
import sys
import os
from pathlib import Path

# 把项目根目录加入 `sys.path`。
# 这样脚本在直接运行时，也能导入仓库里的 `rosetta.*` 模块。
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from transformers import AutoModelForCausalLM, AutoTokenizer
from rosetta.model.wrapper import RosettaModel
from rosetta.model.aligner import TokenAligner, AlignmentStrategy
from rosetta.model.projector import AllInOneProjector
from rosetta.train.dataset_adapters import generate_kv_cache_index
from typing import Dict, Any, List, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

from rosetta.model.projector import load_projector
from rosetta.model.wrapper import RosettaModel
from rosetta.utils.evaluate import set_default_chat_template
import re

def test_token_aligner(slm_tokenizer: AutoTokenizer, llm_tokenizer: AutoTokenizer):
    """测试 TokenAligner 的行为。

    Args:
        slm_tokenizer: SLM tokenizer
        llm_tokenizer: LLM tokenizer
    """
    print("\n" + "="*80)
    print("Testing TokenAligner")
    print("="*80)
    
    # FIRST: 当一个 SLM token 对应到多个 LLM token 时，优先选第一个。
    aligner_first = TokenAligner(
        slm_tokenizer=slm_tokenizer,
        llm_tokenizer=llm_tokenizer,
        strategy=AlignmentStrategy.FIRST,
        verbose=True
    )
    
    # LONGEST: 当存在多个候选映射时，优先选择更长的 token 对齐结果。
    aligner_longest = TokenAligner(
        slm_tokenizer=slm_tokenizer,
        llm_tokenizer=llm_tokenizer,
        strategy=AlignmentStrategy.LONGEST,
        verbose=True
    )
    
    # 这里准备几类不同风格的文本，方便快速观察不同 tokenizer 在英文、
    # 中文、emoji 等输入上的切分差异。
    test_texts = [
        "Hello world!",
        "The future of artificial intelligence is",
        "北京是中国的首都",  # Chinese text
        "🚀 Emojis and special characters!",
    ]
    
    for text in test_texts:
        print(f"\nTest text: '{text}'")
        print("-" * 40)
        
        # 可视化 FIRST 策略下的 token 对齐结果。
        print("\nFIRST Strategy:")
        aligner_first.visualize_alignment(text)
        
        # 可视化 LONGEST 策略下的 token 对齐结果。
        print("\nLONGEST Strategy:")
        aligner_longest.visualize_alignment(text)
    
    # 再做一次不带可视化的快速测试，直接打印真实返回值，
    # 便于后续在代码里复用 `align_sequence` 的输出格式。
    sample_text = "This is a test."
    slm_tokens, aligned_llm_tokens = aligner_first.align_sequence(sample_text)
    print(f"\nQuick alignment test for: '{sample_text}'")
    print(f"SLM tokens: {slm_tokens}")
    print(f"Aligned LLM tokens: {aligned_llm_tokens}")
    
    print("\n✅ TokenAligner test completed")
    

def run_inference_example(rosetta_model: RosettaModel, tokenizer: AutoTokenizer, prompt: str):
    """运行一次 Rosetta 推理示例。

    Args:
        rosetta_model: 已经加载完成的 RosettaModel
        tokenizer: 这里默认使用 base model 对应的 tokenizer
        prompt: 聊天消息列表或能被 chat template 接受的输入
    """
    print("Running inference example...")

    device = rosetta_model.device
    
    # 先把消息列表套用 chat template，得到模型最终看到的字符串。
    # `add_generation_prompt=True` 会在结尾补上 assistant 起始标记，
    # 告诉模型“接下来该你生成回答了”。
    # `enable_thinking=False` 表示这里走普通回答格式，而不是额外的思维链模板。
    input_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    print(f"Input text: {input_text}")

    # 再把格式化后的文本编码成张量，并移动到 Rosetta 所在设备。
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    print(f"Input tokens: {inputs['input_ids']}")

    # 这里构造一个最简版的 kv_cache_index。
    #
    # `instruction_index` 对应 prompt 部分的每个位置，重复到整个输入长度减 1。
    # 通常可理解为：这些 token 在 Rosetta 推理时应被视为“指令上下文”。
    #
    # `label_index` 只有一个位置，用来标记最后一个 token 之后的生成起点。
    #
    # 最终把二者作为列表传给 Rosetta，用于控制内部 kv cache 的拼接/读取逻辑。
    # 这个写法比较适合演示最小可运行流程，不是唯一写法。
    instruction_index = torch.tensor([1, 0], dtype=torch.long).repeat(inputs['input_ids'].shape[1] - 1, 1).unsqueeze(0).to(device)
    label_index = torch.tensor([-1, 0], dtype=torch.long).repeat(1, 1).unsqueeze(0).to(device)
    kv_cache_index = [instruction_index, label_index]

    # 下面这一大段被注释掉的代码，是另一条更细粒度的实验路径：
    # 先显式对齐 SLM / LLM token，再手工构造 mask、message mask 和分段 kv cache。
    # 如果后续要排查对齐问题、prefill 行为或 message 边界问题，可以从这里恢复。
    # slm_tokenizer = tokenizer
    # llm_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct") 
    # strategy = "first"
    # aligner = TokenAligner(slm_tokenizer=slm_tokenizer, llm_tokenizer=llm_tokenizer, strategy=AlignmentStrategy(strategy))
    # messages = [
    #     {"role": "user", "content": prompt}
    # ]
    # details = aligner.align_chat_messages(messages, add_generation_prompt=True, return_details=True)
    # slm_ids = torch.tensor(details['slm_ids_padded']).unsqueeze(0)
    # llm_ids = torch.tensor(details['llm_ids_padded']).unsqueeze(0)

    # slm_pad_mask = torch.tensor(details['slm_padding_mask']).unsqueeze(0)
    # llm_pad_mask = torch.tensor(details['llm_padding_mask']).unsqueeze(0)

    # slm_attention_mask = (~slm_pad_mask).float()
    # llm_attention_mask = (~llm_pad_mask).float()

    # message_mask = torch.tensor(details['message_mask'])
    # kv_cache_index = generate_kv_cache_index(slm_ids.shape[1], slm_ids.shape[1])
    # kv_cache_index[~message_mask] = torch.tensor([[-1,0]])

    # kv_idx = kv_cache_index
    # change_points = [0]
    # for i in range(1, kv_idx.size(0)):
    #     if not torch.equal(kv_idx[i], kv_idx[i - 1]):
    #         change_points.append(i)
    # change_points.append(kv_idx.size(0))

    # kv_cache_list = []

    # for i in range(len(change_points) - 1):
    #     start = change_points[i]
    #     end = change_points[i + 1]
    #     kv_cache_list.append(kv_idx[start:end, :].unsqueeze(0).to(device))
    # prefill_kv_cache_list = kv_cache_list[:-1]
    # print(f"Input prompt: '{prompt}'")
    # print(f"Input shape: {slm_ids.shape}")
    # print(f"Device: {device}")
    
    # slm_ids = slm_ids.to(device)
    # llm_ids = llm_ids.to(device)
    # slm_attention_mask = slm_attention_mask.to(device)
    # llm_attention_mask = llm_attention_mask.to(device)

    # 下面这一段是“直接 forward 看 logits”的调试入口，目前保留但默认不执行。
    # 在只想看下一 token 预测，而不是完整生成时，这种方式更适合排查问题。
    # with torch.no_grad():
    #     # outputs = rosetta_model.forward(
    #     #     input_ids=[slm_ids, llm_ids],
    #     #     attention_mask=[slm_attention_mask, llm_attention_mask],
    #     #     kv_cache_index=kv_cache_list,
    #     #     position_ids=torch.arange(slm_ids.shape[1]).unsqueeze(0).to(device),
    #     #     use_cache=True,
    #     #     output_attentions=False,
    #     #     output_hidden_states=False,
    #     #     sample=False,
    #     # )
    #     outputs = rosetta_model(**inputs, kv_cache_index=kv_cache_index)
        
    #     # Get logits and generate next token
    #     logits = outputs.logits
    #     next_token_logits = logits[0, -1, :]
    #     next_token_id = torch.argmax(next_token_logits, dim=-1)
    #     next_token = tokenizer.decode(next_token_id)
        
    #     print(f"Output logits shape: {logits.shape}")
    #     print(f"Next predicted token: '{next_token}'")
    #     print("✅ Inference completed successfully")

    # 1. 使用 RosettaModel 生成。
    # 这是主流程：Rosetta 会结合 base model、teacher model 和 projector 一起工作。
    with torch.no_grad():
        # outputs = rosetta_model.generate(
        #     prefill_kv_cache_index=prefill_kv_cache_list,
        #     input_ids=[slm_ids, llm_ids],
        #     attention_mask=[slm_attention_mask, llm_attention_mask],
        #     use_cache=True,
        #     output_attentions=False,
        #     output_hidden_states=False,
        #     max_new_tokens=256,
        #     do_sample=False,
        # )
        sampling_params = {
            'do_sample': True,
            'temperature': 0.7,
            'top_p': 0.8,
            'top_k': 20,
            'min_p': 0.0,
            'repetition_penalty': 1.2,
            'max_new_tokens': 1024
        }
        outputs = rosetta_model.generate(**inputs, kv_cache_index=kv_cache_index, **sampling_params)
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Rosetta output text: {output_text}")
    
    # 2. 单独跑 SLM。
    # 这样可以直接对比：Rosetta 的输出相对 base model 本身有没有收益或偏移。
    with torch.no_grad():
        sampling_params = {
            'do_sample': True,
            'temperature': 0.7,
            'top_p': 0.8,
            'top_k': 20,
            'min_p': 0.0,
            'repetition_penalty': 1.2,
            'max_new_tokens': 1024
        }
        slm_model = rosetta_model.model_list[0]
        outputs = slm_model.generate(**inputs, **sampling_params)
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"SLM output text: {output_text}")
    
    # 3. 单独跑 LLM。
    # 这一步有助于判断 teacher 的生成风格，以及 Rosetta 最终输出是否更接近 teacher。
    with torch.no_grad():
        sampling_params = {
            'do_sample': True,
            'temperature': 0.7,
            'top_p': 0.8,
            'top_k': 20,
            'min_p': 0.0,
            'repetition_penalty': 1.2,
            'max_new_tokens': 1024
        }
        llm_model = rosetta_model.model_list[1]
        outputs = llm_model.generate(**inputs, **sampling_params)
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"LLM output text: {output_text}")

def load_rosetta_model(model_config: Dict[str, Any], eval_config: Dict[str, Any], 
                      device: torch.device) -> Tuple[Any, Any]:
    """
    加载带 projector 的 RosettaModel。
    
    Args:
        model_config: 模型配置字典，主要读取 `rosetta_config`
        eval_config: 评估配置字典，保留给旧配置兼容使用
        device: 模型要加载到的设备
        
    Returns:
        Tuple of `(rosetta_model, tokenizer)`
    """
    # 优先从 `model.rosetta_config.checkpoints_dir` 读取 projector checkpoint，
    # 只有旧配置没有迁移时，才回退到 `eval_config.checkpoints_dir`。
    rosetta_config = model_config["rosetta_config"]
    checkpoint_dir = rosetta_config.get("checkpoints_dir", eval_config.get("checkpoints_dir"))
    if checkpoint_dir is None:
        raise KeyError("checkpoints_dir must be provided under model.rosetta_config (preferred) or eval config (legacy)")
    slm_model_path = rosetta_config["base_model"]
    llm_model_path = rosetta_config["teacher_model"]

    # tokenizer 以 base model 为准，因为推理示例里默认从 base model 的输入空间出发。
    slm_tokenizer = AutoTokenizer.from_pretrained(str(slm_model_path))
    set_default_chat_template(slm_tokenizer, slm_model_path)
    
    # 分别加载 SLM 和 teacher LLM，并切到 eval 模式，避免 dropout 等训练行为干扰推理。
    # `torch_dtype=torch.bfloat16` 是为了降低显存占用，同时保持较好的数值稳定性。
    slm_model = AutoModelForCausalLM.from_pretrained(
        str(slm_model_path),
        torch_dtype=torch.bfloat16,
        device_map={"": device}
    ).eval()
    
    llm_model = AutoModelForCausalLM.from_pretrained(
        str(llm_model_path),
        torch_dtype=torch.bfloat16,
        device_map={"": device}
    ).eval()
    
    # 自动扫描 checkpoint 目录中形如 `projector_0.pt`, `projector_1.pt` 的文件，
    # 从而推断当前有多少个 projector 需要加载。
    num_projectors = len([f for f in os.listdir(checkpoint_dir) if re.match(r"projector_\d+\.pt", f)])
    projector_list = []
    for t in range(num_projectors):
        # 先根据 json 配置实例化 projector 结构。
        json_cfg = os.path.join(checkpoint_dir, f"projector_{t}.json")
        proj = load_projector(json_cfg)
        proj = proj.to(device)

        # 再加载对应权重；`strict=False` 让它对少量配置差异更宽容，
        # 适合实验阶段快速验证 checkpoint 是否可用。
        pt_path = os.path.join(checkpoint_dir, f"projector_{t}.pt")
        if os.path.exists(pt_path):
            state_dict = torch.load(pt_path, map_location=device)
            proj.load_state_dict(state_dict, strict=False)
        projector_list.append(proj)
    
    # 组装 RosettaModel。
    # `base_model_idx=0` 表示 `model_list[0]` 是主模型，也就是这里的 SLM。
    rosetta_model = RosettaModel(
        model_list=[slm_model, llm_model],
        base_model_idx=0,
        projector_list=projector_list,
    ).to(device).eval()

    # projector 之外通常还需要加载 projector 的映射配置，
    # 用来告诉 Rosetta 在哪些层、哪些位置应用这些 projector。
    proj_cfg_path = os.path.join(checkpoint_dir, "projector_config.json")
    rosetta_model.load_projector_config(proj_cfg_path)

    return rosetta_model, slm_tokenizer

def main():
    """脚本入口：加载模型并运行一次示例推理。"""

    # 这里直接写死了一个本地可运行的最小示例配置，
    # 方便开发时用 `python inference_example.py` 立即验证整条链路。
    rosetta_model, slm_tokenizer = load_rosetta_model(
        model_config={
            "rosetta_config": {
                "base_model": "Qwen/Qwen3-0.6B",
                "teacher_model": "Qwen/Qwen3-4B",
                "checkpoints_dir": "local/checkpoints/0.6B_4B_general/final"
            }
        },
        eval_config={},
        device=torch.device("cuda")
    )
    
    # 如需调试 tokenizer 对齐行为，可取消下面这行注释。
    # test_token_aligner(slm_tokenizer, llm_tokenizer)
    
    # 示例 prompt 采用聊天消息格式，与上面的 `apply_chat_template` 相匹配。
    prompt = [{
        "role": "user",
        "content": "Accurately answer the following question:\n\nStatement 1 | If T: V -> W is a linear transformation and dim(V ) < dim(W) < 1, then T must be injective. Statement 2 | Let dim(V) = n and suppose that T: V -> V is linear. If T is injective, then it is a bijection.\n\nAre these statements correct? Let's think step by step and then answer the question starting with Answer:"
    }]
    run_inference_example(rosetta_model, slm_tokenizer, prompt)
    # 也可以直接换成中文 prompt 做快速试验，例如下面这一行：
    # run_inference_example(rosetta_model, slm_tokenizer, "从美国向北进入加拿大时，您会看到北星（北极星）越来越")


if __name__ == "__main__":
    # 如需远程调试，可打开下面的 debugpy 代码。
    # import debugpy
    # debugpy.listen(("0.0.0.0", 5678))
    # print("Waiting for debugger attach...")
    # debugpy.wait_for_client()
    # print("Debugger attached, running...")
    main()
