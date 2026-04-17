#!/usr/bin/env python3
"""
ComfyUI 三视图工作流执行脚本
用法: python3 run_triple_view.py [角色名]
默认角色: 通天教主
"""

import urllib.request
import urllib.error
import json
import uuid
import time
import sys
import os

# ============ 配置区 ============
COMFYUI_HOST = "jq1.9gpu.com"
COMFYUI_PORT = 12703
COMFYUI_URL = f"http://{COMFYUI_HOST}:{COMFYUI_PORT}"

# 角色名（默认通天教主）
CHAR_NAME = sys.argv[1] if len(sys.argv) > 1 else "通天教主"

# 角色正面图文件名（需要先上传到 ComfyUI input 目录）
FRONT_IMAGE = f"{CHAR_NAME}_正面.png"
# 参考图（用于提取深度，临时用正面图代替）
REF_IMAGE = f"{CHAR_NAME}_正面.png"

# 输出文件名
OUTPUT_PREFIX = f"{CHAR_NAME}_三视图"

# ============ 提示词 ============
SIDE_PROMPT = f"""Same male Taoist warrior deity, {CHAR_NAME}, square jawed face, sharp sword-shaped brows, black hair in jade pin, wearing white battle robes. Side profile view, showing left side of face, one eye visible. Anime style, Chinese xianxia aesthetic, clean lineart."""

BACK_PROMPT = f"""Same male Taoist warrior deity, {CHAR_NAME}, square jawed face, sharp sword-shaped brows, black hair in jade pin, wearing white battle robes. Back view, showing back of head, hair in jade pin, robes flowing behind. Anime style, Chinese xianxia aesthetic, clean lineart."""

# ============ 工作流节点定义 ============
# 正面→侧面
workflow_side = {
    "1": {
        "inputs": {"image": FRONT_IMAGE},
        "class_type": "LoadImage"
    },
    "2": {
        "inputs": {"pixels": ["1", 0], "vae": ["4", 2]},
        "class_type": "VAEEncode"
    },
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
            "t5xxl": SIDE_PROMPT,
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
        "inputs": {"image": REF_IMAGE},
        "class_type": "LoadImage"
    },
    "7": {
        "inputs": {"use_cpu": "false", "midas_type": "DPT_Hybrid", "invert_depth": "false", "image": ["6", 0]},
        "class_type": "MiDaS Depth Approximation"
    },
    "8": {
        "inputs": {"control_net_name": "FLUX.1/InstantX-FLUX1-Dev-Union/diffusion_pytorch_model.safetensors"},
        "class_type": "ControlNetLoader"
    },
    "9": {
        "inputs": {"type": "depth", "control_net": ["8", 0]},
        "class_type": "SetUnionControlNetType"
    },
    "10": {
        "inputs": {
            "seed": 1234567890,
            "steps": 25,
            "sampler_name": "euler",
            "scheduler": "normal",
            "upscale_by": 1.5,
            "controlnet_strength": 0.7,
            "control_end": 1.0,
            "color_match_strength": 0.3,
            "seed_shift": 0,
            "model": ["3", 0],
            "positive": ["5", 0],
            "vae": ["3", 2],
            "control_net": ["9", 0],
            "image": ["7", 0],
            "latent": ["2", 0]
        },
        "class_type": "FluxControlnetSampler"
    },
    "11": {
        "inputs": {"samples": ["10", 1], "vae": ["3", 2]},
        "class_type": "VAEDecode"
    },
    "12": {
        "inputs": {"filename_prefix": f"{OUTPUT_PREFIX}_侧面", "images": ["11", 0]},
        "class_type": "SaveImage"
    }
}

# 正面→背面
workflow_back = {
    "1": {
        "inputs": {"image": FRONT_IMAGE},
        "class_type": "LoadImage"
    },
    "2": {
        "inputs": {"pixels": ["1", 0], "vae": ["4", 2]},
        "class_type": "VAEEncode"
    },
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
            "t5xxl": BACK_PROMPT,
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
        "inputs": {"image": REF_IMAGE},
        "class_type": "LoadImage"
    },
    "7": {
        "inputs": {"use_cpu": "false", "midas_type": "DPT_Hybrid", "invert_depth": "false", "image": ["6", 0]},
        "class_type": "MiDaS Depth Approximation"
    },
    "8": {
        "inputs": {"control_net_name": "FLUX.1/InstantX-FLUX1-Dev-Union/diffusion_pytorch_model.safetensors"},
        "class_type": "ControlNetLoader"
    },
    "9": {
        "inputs": {"type": "depth", "control_net": ["8", 0]},
        "class_type": "SetUnionControlNetType"
    },
    "10": {
        "inputs": {
            "seed": 9876543210,
            "steps": 25,
            "sampler_name": "euler",
            "scheduler": "normal",
            "upscale_by": 1.5,
            "controlnet_strength": 0.7,
            "control_end": 1.0,
            "color_match_strength": 0.3,
            "seed_shift": 0,
            "model": ["3", 0],
            "positive": ["5", 0],
            "vae": ["3", 2],
            "control_net": ["9", 0],
            "image": ["7", 0],
            "latent": ["2", 0]
        },
        "class_type": "FluxControlnetSampler"
    },
    "11": {
        "inputs": {"samples": ["10", 1], "vae": ["3", 2]},
        "class_type": "VAEDecode"
    },
    "12": {
        "inputs": {"filename_prefix": f"{OUTPUT_PREFIX}_背面", "images": ["11", 0]},
        "class_type": "SaveImage"
    }
}


def queue_prompt(workflow, name="job"):
    """通过 REST API 提交工作流"""
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
    
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
            print(f"✅ [{name}] 提交成功! prompt_id: {result.get('prompt_id', 'N/A')}")
            return prompt_id, result
    except urllib.error.HTTPError as e:
        body = e.read()
        print(f"❌ [{name}] HTTP {e.code}: {body[:1000]}")
        return None, None
    except Exception as e:
        print(f"❌ [{name}] Error: {e}")
        return None, None


def check_history(prompt_id, timeout=120):
    """轮询检查工作流状态"""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(f"{COMFYUI_URL}/api/history?per_page=5", timeout=10) as resp:
                history = json.loads(resp.read())
                if prompt_id in history:
                    entry = history[prompt_id]
                    status = entry.get('status', {}).get('status_str', 'unknown')
                    print(f"   状态: {status}")
                    return status
        except Exception as e:
            print(f"   查询历史出错: {e}")
        time.sleep(2)
    return "timeout"


def main():
    print(f"=" * 50)
    print(f"ComfyUI 三视图生成")
    print(f"角色: {CHAR_NAME}")
    print(f"输出前缀: {OUTPUT_PREFIX}")
    print(f"=" * 50)
    
    # 检查图片是否存在
    print(f"\n检查图片: {FRONT_IMAGE}")
    
    # 提交侧面图工作流
    print(f"\n{'='*50}")
    print(f"第1步: 生成侧面图...")
    pid1, res1 = queue_prompt(workflow_side, "侧面")
    
    if pid1:
        print("等待生成完成 (约30秒)...")
        time.sleep(40)
        status1 = check_history(pid1)
        if status1 == "success":
            print("✅ 侧面图生成成功!")
        else:
            print(f"⚠️ 侧面图状态: {status1}")
    
    # 提交背面图工作流
    print(f"\n{'='*50}")
    print(f"第2步: 生成背面图...")
    pid2, res2 = queue_prompt(workflow_back, "背面")
    
    if pid2:
        print("等待生成完成 (约30秒)...")
        time.sleep(40)
        status2 = check_history(pid2)
        if status2 == "success":
            print("✅ 背面图生成成功!")
        else:
            print(f"⚠️ 背面图状态: {status2}")
    
    print(f"\n{'='*50}")
    print("全部完成!")
    print(f"图片保存在 ComfyUI output 目录:")
    print(f"  - {OUTPUT_PREFIX}_侧面_00001_.png")
    print(f"  - {OUTPUT_PREFIX}_背面_00001_.png")


if __name__ == "__main__":
    main()
