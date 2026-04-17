#!/usr/bin/env python3
"""
SDXL + InstantID 三视图生成脚本
正面图 → IP-Adapter FaceID + instantid ControlNet → 侧面/背面/正面

2026-04-17
"""

import urllib.request
import json
import uuid
import time
import random
import paramiko
import os
import sys

COMFYUI_URL = "http://jq1.9gpu.com:12703"

def submit_prompt(workflow):
    prompt_id = str(uuid.uuid4())
    data = {"prompt": workflow, "prompt_id": prompt_id, "extra_data": {}}
    body = json.dumps(data, ensure_ascii=False).encode('utf-8')
    req = urllib.request.Request(
        f"{COMFYUI_URL}/api/prompt",
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())
        return result.get("prompt_id"), result.get("number")

def wait_for_done(prompt_id, timeout=180):
    start = time.time()
    while time.time() - start < timeout:
        time.sleep(10)
        try:
            with urllib.request.urlopen(f"{COMFYUI_URL}/api/history/{prompt_id}", timeout=15) as r:
                entry = json.loads(r.read()).get(prompt_id, {})
                status = entry.get('status', {}).get('status_str', 'unknown')
                if status in ('success', 'error'):
                    return status, entry
        except:
            pass
    return 'timeout', {}

def download_result(prompt_id, node_id, save_dir="/home/rs8568/.openclaw/agents/xiaolin/workspace/triple_test"):
    os.makedirs(save_dir, exist_ok=True)
    with urllib.request.urlopen(f"{COMFYUI_URL}/api/history/{prompt_id}", timeout=15) as r:
        entry = json.loads(r.read()).get(prompt_id, {})
        outputs = entry.get('outputs', {})
        if node_id not in outputs:
            return None, entry
        imgs = outputs[node_id].get('images', [])
        if not imgs:
            return None, entry
        
        img = imgs[0]
        remote_path = f"/root/ComfyUI/output/{img['filename']}"
        local_path = os.path.join(save_dir, img['filename'])
        
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect('jq1.9gpu.com', port=12700, username='root', password='D~Bu*~se', timeout=15)
        sftp = client.open_sftp()
        sftp.get(remote_path, local_path)
        sftp.close()
        client.close()
        return local_path, entry

def build_workflow(char_name, input_image, angle_prompt, side_prefix, seed):
    """构建SDXL InstantID三视图工作流"""
    workflow = {
        # 1. SDXL CheckpointLoader
        "3": {
            "inputs": {"ckpt_name": "sd_xl_base_1.0_0.9vae.safetensors"},
            "class_type": "CheckpointLoaderSimple"
        },
        # 2. LoadImage - 人脸输入图
        "2": {
            "inputs": {"image": input_image},
            "class_type": "LoadImage"
        },
        # 3. IPAdapterInsightFaceLoader
        "15": {
            "inputs": {"provider": "CUDA"},
            "class_type": "IPAdapterInsightFaceLoader"
        },
        # 4. easy pipeIn - 创建空的pipe_line
        "14": {
            "inputs": {
                "pipe": None,
                "model": ["3", 0],  # MODEL
                "pos": None,
                "neg": None,
                "latent": None,
                "vae": ["3", 2]  # VAE
            },
            "class_type": "easy pipeIn"
        },
        # 5. easy instantIDApply - 核心节点！
        "10": {
            "inputs": {
                "pipe": ["14", 0],  # PIPE_LINE
                "image": ["2", 0],  # 人脸图
                "instantid_file": "SDXL/ip-adapter-faceid-plusv2_sdxl.bin",
                "insightface": ["15", 0],  # INSIGHTFACE
                "control_net_name": "SDXL/instantid/diffusion_pytorch_model.safetensors",
                "cn_strength": 0.8,
                "cn_soft_weights": 0.7,
                "weight": 1.0,
                "start_at": 0.0,
                "end_at": 1.0,
                "noise": 0.0
            },
            "class_type": "easy instantIDApply"
        },
        # 6. FromBasicPipe - 从pipe取出处理后的model和conditioning
        "11": {
            "inputs": {"basic_pipe": ["10", 0]},  # PIPE_LINE
            "class_type": "FromBasicPipe"
        },
        # 7. CLIPTextEncodeSDXL - 正面提示词
        "4": {
            "inputs": {
                "clip": ["3", 1],  # CLIP
                "width": 768,
                "height": 1344,
                "crop_w": 0,
                "crop_h": 0,
                "target_width": 768,
                "target_height": 1344,
                "text_g": angle_prompt,
                "text_l": angle_prompt
            },
            "class_type": "CLIPTextEncodeSDXL"
        },
        # 8. CLIPTextEncodeSDXL - 负面提示词
        "5": {
            "inputs": {
                "clip": ["3", 1],
                "width": 768,
                "height": 1344,
                "crop_w": 0,
                "crop_h": 0,
                "target_width": 768,
                "target_height": 1344,
                "text_g": "blurry, low quality, bad anatomy, extra fingers, distorted face, anime style contamination, cartoon, chibi",
                "text_l": "blurry, low quality, bad anatomy, extra fingers, distorted face"
            },
            "class_type": "CLIPTextEncodeSDXL"
        },
        # 9. KSampler - 采样
        "6": {
            "inputs": {
                "model": ["11", 0],  # 处理后的MODEL
                "seed": seed,
                "steps": 30,
                "cfg": 7.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "positive": ["11", 3],  # FromBasicPipe[3]=positive CONDITIONING
                "negative": ["11", 4],   # FromBasicPipe[4]=negative CONDITIONING
                "latent_image": {"latent": {"samples": [[1, 4, 84, 144]]}},  # 空latent，768x1344
                "denoise": 0.7
            },
            "class_type": "KSampler"
        },
        # 10. VAEDecode
        "7": {
            "inputs": {"samples": ["6", 0], "vae": ["3", 2]},
            "class_type": "VAEDecode"
        },
        # 11. SaveImage
        "8": {
            "inputs": {"filename_prefix": f"SDXL_{char_name}_{side_prefix}", "images": ["7", 0]},
            "class_type": "SaveImage"
        }
    }
    return workflow

# ========== 测试SDXL InstantID工作流 ==========
print("=" * 60)
print("SDXL + InstantID 三视图生成测试")
print("=" * 60)

char_name = "通天教主"
input_image = "通天教主_正面.png"

# 三个角度的提示词
angles = {
    "正面": "Full front view, same male Taoist warrior deity facing camera, square jawed face, sharp sword-shaped brows, black hair in jade pin, white battle robes. Anime style, Chinese xianxia aesthetic, clean lineart.",
    "侧面左": "Left side profile view, same male Taoist warrior deity, showing left half of face, one eye visible, square jaw, sharp brows, black hair in jade crown, white robes. Anime style, Chinese xianxia aesthetic, clean lineart.",
    "背面": "Back view, same male Taoist warrior deity facing away, black hair in jade pin, white battle robes, broad shoulders visible. Anime style, Chinese xianxia aesthetic, clean lineart.",
}

results = {}
for angle_name, angle_prompt in angles.items():
    seed = random.randint(1, 999999999)
    print(f"\n>>> 生成 {char_name} {angle_name}...")
    
    workflow = build_workflow(char_name, input_image, angle_prompt, angle_name, seed)
    prompt_id, number = submit_prompt(workflow)
    print(f"    提交成功: prompt_id={prompt_id}, number={number}")
    
    status, entry = wait_for_done(prompt_id, timeout=180)
    print(f"    状态: {status}")
    
    if status == 'success':
        local_path, _ = download_result(prompt_id, '8')
        if local_path:
            print(f"    ✅ 下载: {local_path}")
            results[angle_name] = local_path
        else:
            print(f"    ⚠️ 无输出图片")
            # 打印输出节点
            outputs = entry.get('outputs', {})
            print(f"    输出节点: {list(outputs.keys())}")
    else:
        msgs = entry.get('status', {}).get('messages', [])
        for m in msgs:
            if m[0] == 'execution_error':
                print(f"    ❌ 错误: {m[1].get('exception_message', '')[:300]}")
    
    time.sleep(3)

print("\n" + "=" * 60)
print("生成结果:")
for angle, path in results.items():
    print(f"  {angle}: {path}")
print("=" * 60)
