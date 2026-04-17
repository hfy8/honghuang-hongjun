#!/usr/bin/env python3
"""
三视图生成测试脚本
测试两种策略:
A) 纯FLUX Img2Img - 从正面图重绘，测试能否改变角度
B) 纯文本生成 - 不依赖输入图，直接生成各角度立绘

2026-04-17
"""

import urllib.request
import urllib.error
import json
import uuid
import time
import random
import paramiko
import os
import sys

COMFYUI_URL = "http://jq1.9gpu.com:12703"
REMOTE_HOST = "jq1.9gpu.com"
REMOTE_PORT = 12700
REMOTE_USER = "root"
REMOTE_PASS = "D~Bu*~se"

def submit_prompt(workflow, prompt_id=None):
    """提交工作流到ComfyUI"""
    if prompt_id is None:
        prompt_id = str(uuid.uuid4())
    data = {
        "prompt": workflow,
        "prompt_id": prompt_id,
        "extra_data": {}
    }
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

def wait_and_check(prompt_id, timeout=90):
    """等待并检查结果"""
    start = time.time()
    while time.time() - start < timeout:
        time.sleep(8)
        try:
            with urllib.request.urlopen(f"{COMFYUI_URL}/api/history/{prompt_id}", timeout=15) as r:
                entry = json.loads(r.read()).get(prompt_id, {})
                status = entry.get('status', {}).get('status_str', 'unknown')
                outputs = entry.get('outputs', {})
                if status in ('success', 'error'):
                    return status, outputs, entry.get('status', {}).get('messages', [])
        except Exception as e:
            print(f"  检查出错: {e}")
    return 'timeout', {}, []

def download_image(remote_path, local_path):
    """下载ComfyUI生成的图片"""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(REMOTE_HOST, port=REMOTE_PORT, username=REMOTE_USER, password=REMOTE_PASS, timeout=15)
    sftp = client.open_sftp()
    sftp.get(remote_path, local_path)
    sftp.close()
    client.close()

def save_result(prompt_id, node_id, local_dir="/tmp/triple_test"):
    """下载最新生成的结果图片"""
    os.makedirs(local_dir, exist_ok=True)
    with urllib.request.urlopen(f"{COMFYUI_URL}/api/history/{prompt_id}", timeout=15) as r:
        entry = json.loads(r.read()).get(prompt_id, {})
        outputs = entry.get('outputs', {})
        if node_id in outputs:
            imgs = outputs[node_id].get('images', [])
            for img in imgs:
                remote_path = f"/root/ComfyUI/output/{img['filename']}"
                local_path = os.path.join(local_dir, img['filename'])
                try:
                    download_image(remote_path, local_path)
                    print(f"  下载: {img['filename']} → {local_path}")
                    return local_path
                except Exception as e:
                    print(f"  下载失败 {img['filename']}: {e}")
    return None

# ========== 策略A: 纯FLUX Img2Img 不同denoise测试 ==========
print("=" * 60)
print("策略A: FLUX Img2Img 不同denoise值测试")
print("=" * 60)

for denoise in [0.5, 0.65, 0.8]:
    for seed_offset in range(2):
        seed = random.randint(1, 999999999999999)
        prompt_id = str(uuid.uuid4())
        
        # 纯Img2Img，从正面图出发
        workflow = {
            "3": {
                "inputs": {
                    "model_name": "FLUX1/flux1-dev-fp8.safetensors",
                    "weight_dtype": "fp8_e4m3fn",
                    "clip_name1": "t5/t5xxl_fp8_e4m3fn_scaled.safetensors",
                    "clip_name2_opt": ".none",
                    "vae_name": "flux-ae.safetensors",
                    "clip_vision_name": ".none",
                    "style_model_name": ".none"
                },
                "class_type": "FluxLoader"
            },
            "1": {
                "inputs": {"image": "通天教主_正面.png"},
                "class_type": "LoadImage"
            },
            "2": {
                "inputs": {"pixels": ["1", 0], "vae": ["3", 2]},
                "class_type": "VAEEncode"
            },
            "4": {
                "inputs": {
                    "clip_l": "",
                    "t5xxl": "Same male Taoist warrior deity, square jawed face, sharp sword-shaped brows, black hair in jade pin, wearing white battle robes. Side profile view, left side of face visible, one eye. Anime style, Chinese xianxia aesthetic, clean lineart.",
                    "guidance": 3.5,
                    "clip": ["3", 1]
                },
                "class_type": "CLIPTextEncodeFlux"
            },
            "5": {
                "inputs": {"guidance": 3.5, "conditioning": ["4", 0]},
                "class_type": "FluxGuidance"
            },
            "6": {
                "inputs": {
                    "seed": seed,
                    "steps": 25,
                    "cfg": 1.0,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": denoise,
                    "input_type": "latent",
                    "x_slices": 1,
                    "y_slices": 1,
                    "overlap": 0.25,
                    "batch_size": 1,
                    "use_sliced_conditioning": True,
                    "conditioning_strength": 1.2,
                    "debug_mode": False,
                    "model": ["3", 0],
                    "positive": ["5", 0],
                    "negative": ["5", 0],
                    "latent_image": ["2", 0],
                    "vae": ["3", 2]
                },
                "class_type": "FL_KsamplerPlusV2"
            },
            "7": {
                "inputs": {"samples": ["6", 3], "vae": ["3", 2]},
                "class_type": "VAEDecode"
            },
            "8": {
                "inputs": {"filename_prefix": f"A_denoise{int(denoise*100)}_s{seed_offset}", "images": ["7", 0]},
                "class_type": "SaveImage"
            }
        }

        print(f"\n→ 测试A: denoise={denoise}, seed_offset={seed_offset}, seed={seed}")
        pid, num = submit_prompt(workflow, prompt_id)
        print(f"  提交成功: prompt_id={pid}, number={num}")
        
        status, outputs, msgs = wait_and_check(pid, timeout=90)
        print(f"  状态: {status}")
        
        if status == 'success' and '8' in outputs:
            local = save_result(pid, '8', "/tmp/triple_test")
            if local:
                print(f"  ✅ 保存: {local}")
        elif status == 'error':
            err_msg = ' '.join([m[1].get('exception_message', '') for m in msgs if m[0] == 'execution_error'])
            print(f"  ❌ 错误: {err_msg[:200]}")
        else:
            print(f"  ⚠️ 超时或无输出")
        
        time.sleep(3)

# ========== 策略B: 纯文本生成（无输入图） ==========
print("\n" + "=" * 60)
print("策略B: 纯文本生成测试（无输入图）")
print("=" * 60)

angle_prompts = {
    "front": "Full front view, male Taoist warrior deity facing camera directly, square jawed face, sharp sword-shaped brows, black hair in jade pin, wearing white battle robes. Anime style, Chinese xianxia aesthetic, clean lineart.",
    "side_left": "Left side profile view, male Taoist warrior deity, showing left half of face, one eye visible, square jaw, sharp brows, black hair in jade crown, white robes. Anime style, Chinese xianxia aesthetic, clean lineart.",
    "side_right": "Right side profile view, male Taoist warrior deity, showing right half of face, one eye visible, square jaw, sharp brows, black hair in jade crown, white robes. Anime style, Chinese xianxia aesthetic, clean lineart.",
    "back": "Back view, male Taoist warrior deity facing away from camera, black hair in jade pin, white battle robes, broad shoulders. Anime style, Chinese xianxia aesthetic, clean lineart.",
}

for angle_name, prompt_text in angle_prompts.items():
    seed = random.randint(1, 999999999999999)
    prompt_id = str(uuid.uuid4())
    
    # 纯文本生成，无VAEEncode
    workflow = {
        "3": {
            "inputs": {
                "model_name": "FLUX1/flux1-dev-fp8.safetensors",
                "weight_dtype": "fp8_e4m3fn",
                "clip_name1": "t5/t5xxl_fp8_e4m3fn_scaled.safetensors",
                "clip_name2_opt": ".none",
                "vae_name": "flux-ae.safetensors",
                "clip_vision_name": ".none",
                "style_model_name": ".none"
            },
            "class_type": "FluxLoader"
        },
        "4": {
            "inputs": {
                "clip_l": "",
                "t5xxl": prompt_text,
                "guidance": 3.5,
                "clip": ["3", 1]
            },
            "class_type": "CLIPTextEncodeFlux"
        },
        "5": {
            "inputs": {"guidance": 3.5, "conditioning": ["4", 0]},
            "class_type": "FluxGuidance"
        },
        "6": {
            "inputs": {
                "seed": seed,
                "steps": 30,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "input_type": "latent",
                "x_slices": 1,
                "y_slices": 1,
                "overlap": 0.25,
                "batch_size": 1,
                "use_sliced_conditioning": True,
                "conditioning_strength": 1.2,
                "debug_mode": False,
                "model": ["3", 0],
                "positive": ["5", 0],
                "negative": ["5", 0],
                "latent_image": [],
                "vae": ["3", 2]
            },
            "class_type": "FL_KsamplerPlusV2"
        },
        "7": {
            "inputs": {"samples": ["6", 3], "vae": ["3", 2]},
            "class_type": "VAEDecode"
        },
        "8": {
            "inputs": {"filename_prefix": f"B_{angle_name}", "images": ["7", 0]},
            "class_type": "SaveImage"
        }
    }

    print(f"\n→ 测试B: {angle_name}")
    pid, num = submit_prompt(workflow, prompt_id)
    print(f"  提交成功: prompt_id={pid}, number={num}")
    
    status, outputs, msgs = wait_and_check(pid, timeout=120)
    print(f"  状态: {status}")
    
    if status == 'success' and '8' in outputs:
        local = save_result(pid, '8', "/tmp/triple_test")
        if local:
            print(f"  ✅ 保存: {local}")
    elif status == 'error':
        err_msg = ' '.join([m[1].get('exception_message', '') for m in msgs if m[0] == 'execution_error'])
        print(f"  ❌ 错误: {err_msg[:200]}")
    else:
        print(f"  ⚠️ 超时或无输出")
    
    time.sleep(3)

print("\n" + "=" * 60)
print("测试完成! 查看 /tmp/triple_test/ 目录下的结果")
print("=" * 60)
