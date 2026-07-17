# GRPO Quick Start Guide for Camera Movement VQA

**Last Updated**: 2026-05-25  
**Status**: Ready to implement

---

## TL;DR - What You Need to Know

1. **Start Here**: `/camera_movement_sft/rl/rewards.py` (need to create)
2. **Reference**: `/swift/rewards/orm.py` (how to extend ORM class)
3. **Example**: `/examples/train/grpo/external/grpo_7b.sh` (full GRPO training)
4. **Launch**: `swift rlhf --rlhf_type grpo --reward_funcs camera_movement_f1 ...`

---

## Implementation Checklist

### Phase 1: Reward Functions (CRITICAL - START HERE)

**Step 1**: Create base reward class
```python
# File: /camera_movement_sft/rl/rewards.py

from swift.rewards import ORM
import json
from typing import List, Dict

class CameraMovementORM(ORM):
    """Base class for camera movement reward functions."""
    
    def __init__(self):
        self.eval_code = None  # Will load evaluation code
        
    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        """
        Args:
            completions: Model outputs (JSON strings)
            **kwargs: May include 'ground_truths', 'prompts', etc.
        
        Returns:
            rewards: Float rewards in [0, 1] range
        """
        rewards = []
        for completion in completions:
            try:
                pred = json.loads(completion)
                gt = kwargs.get('ground_truths', [None])[len(rewards)]
                reward = self.compute_reward(pred, gt)
                rewards.append(reward)
            except Exception as e:
                print(f"Error computing reward: {e}")
                rewards.append(0.0)
        return rewards
    
    def compute_reward(self, pred, gt) -> float:
        """Override in subclasses."""
        raise NotImplementedError
```

**Step 2**: Implement specific reward functions
```python
class CameraMovementF1ORM(CameraMovementORM):
    """Frame-level F1 score reward."""
    
    def compute_reward(self, pred, gt) -> float:
        # Import evaluation logic
        from camera_movement_sft.eval.evaluate_camera_movement_fixed import evaluate_frame_level
        
        # Compute F1 score
        metrics = evaluate_frame_level(gt, pred)  # Returns dict with 'f1' key
        return min(metrics.get('f1', 0.0), 1.0)  # Clamp to [0, 1]

class CameraMovementFormatORM(CameraMovementORM):
    """Reward for valid JSON format."""
    
    def compute_reward(self, pred, gt) -> float:
        try:
            data = json.loads(pred)
            # Check required fields
            if 'segments' in data and isinstance(data['segments'], list):
                # Check each segment
                for seg in data['segments']:
                    if 'basic_movement' in seg and 'start_time' in seg:
                        continue
                    return 0.5  # Partial structure
                return 1.0  # Valid
            return 0.0
        except:
            return 0.0

class CameraMovementCompositeORM(CameraMovementORM):
    """Multi-signal composite reward."""
    
    def compute_reward(self, pred, gt) -> float:
        from camera_movement_sft.eval.evaluate_camera_movement_fixed import evaluate_frame_level
        
        # Frame-level F1 (70%)
        try:
            metrics = evaluate_frame_level(gt, pred)
            frame_f1 = min(metrics.get('f1', 0.0), 1.0)
        except:
            frame_f1 = 0.0
        
        # Format reward (15%)
        format_reward = self.check_format(pred)
        
        # Confidence penalty (15%)
        confidence_reward = self.check_confidence(pred)
        
        # Composite
        return 0.70 * frame_f1 + 0.15 * format_reward + 0.15 * confidence_reward
    
    @staticmethod
    def check_format(pred: str) -> float:
        try:
            data = json.loads(pred)
            if all(k in data for k in ['video_id', 'segments']):
                return 1.0
            return 0.5
        except:
            return 0.0
    
    @staticmethod
    def check_confidence(pred: str) -> float:
        try:
            data = json.loads(pred)
            confidences = []
            for seg in data.get('segments', []):
                conf = seg.get('confidence', 'medium')
                if conf == 'high':
                    confidences.append(1.0)
                elif conf == 'medium':
                    confidences.append(0.8)
                else:
                    confidences.append(0.6)
            return sum(confidences) / len(confidences) if confidences else 0.0
        except:
            return 0.5

# Register with ms-swift
orms = {
    'camera_movement_f1': CameraMovementF1ORM,
    'camera_movement_format': CameraMovementFormatORM,
    'camera_movement_composite': CameraMovementCompositeORM,
}
```

**Step 3**: Register with ms-swift
```bash
# In /swift/rewards/__init__.py, add:
from camera_movement_sft.rl.rewards import orms as camera_orms
orms.update(camera_orms)
```

**Step 4**: Unit tests
```python
# File: /camera_movement_sft/rl/unit_tests_reward.py

import unittest
from rewards import CameraMovementF1ORM, CameraMovementCompositeORM

class TestCameraRewards(unittest.TestCase):
    
    def test_f1_reward(self):
        orm = CameraMovementF1ORM()
        
        # Test perfect prediction
        gt = {
            "segments": [
                {"start_time": 0, "end_time": 2, "basic_movement": [{"type": "Pan", "direction": "left"}]}
            ]
        }
        pred = json.dumps(gt)
        reward = orm(completions=[pred], ground_truths=[gt])[0]
        self.assertGreater(reward, 0.9)
        
        # Test empty prediction
        empty = json.dumps({"segments": []})
        reward = orm(completions=[empty], ground_truths=[gt])[0]
        self.assertLess(reward, 0.1)

if __name__ == '__main__':
    unittest.main()
```

---

### Phase 2: Training Configuration

**Step 1**: Create config file
```yaml
# File: /camera_movement_sft/rl/config_grpo.yaml

# Model & Data
model_id: "qwen/Qwen2.5-7B-Instruct"
dataset_id: "camera_movement_rl"
output_dir: "./output_grpo"

# GRPO Parameters
rlhf_type: "grpo"
num_generations: 4          # Completions per prompt
num_epochs: 3               # Training epochs
learning_rate: 1.0e-6       # Conservative to preserve SFT
beta: 0.04                  # KL divergence weight
max_completion_length: 1024

# Reward Functions
reward_funcs: ["camera_movement_composite", "camera_movement_format"]
reward_weights: [0.85, 0.15]

# Generation (vLLM)
use_vllm: true
vllm_mode: "colocate"       # or "server" if external
vllm_gpu_memory_utilization: 0.6

# Training
batch_size: 4
per_device_train_batch_size: 1
gradient_accumulation_steps: 4
num_train_epochs: 3
logging_steps: 10
save_steps: 100
eval_steps: 50

# Advantage Estimation
advantage_estimator: "grpo"  # or "rloo"
scale_rewards: "group"       # Normalize within group
```

**Step 2**: Create training script (optional)
```bash
# File: /camera_movement_sft/rl/train_grpo.sh

#!/bin/bash

CONFIG="config_grpo.yaml"
MODEL="path/to/sft_model"
DATASET="camera_movement_train"

swift rlhf \
    --rlhf_type grpo \
    --model ${MODEL} \
    --dataset ${DATASET} \
    --reward_funcs camera_movement_composite camera_movement_format \
    --reward_weights 0.85 0.15 \
    --num_generations 4 \
    --learning_rate 1e-6 \
    --output_dir ./output_grpo \
    --batch_size 4 \
    --num_train_epochs 3 \
    --use_vllm true \
    --vllm_mode colocate
```

---

### Phase 3: Documentation

Create these three documents:

1. **GRPO_IMPLEMENTATION_GUIDE.md**
   - Architecture overview
   - Component interaction diagram
   - Full training pipeline

2. **CAMERA_MOVEMENT_REWARD_DESIGN.md**
   - Reward function design rationale
   - 7 options with pros/cons
   - Selection matrix

3. **SFT_TO_RL_BEST_PRACTICES.md**
   - Data preparation
   - Hyperparameter tuning
   - Common pitfalls

---

### Phase 4: Validation

**Small-scale test**:
```bash
# Use subset of data (100-500 examples)
# Run on 1-2 GPUs
# Monitor these metrics:

1. Composite F1: Should increase from SFT baseline
2. Format Compliance: Should stay ~100%
3. Training Loss: Should decrease smoothly
4. Generation Diversity: Should increase
```

**Expected Results**:
- Phase 1 (Rewards): +5-10% F1 improvement
- Phase 2 (Config): Training runs without OOM errors
- Phase 3 (Docs): Complete runnable examples
- Phase 4 (Validation): Confirmed metrics improvement

---

## Key Reference Code

### Extending ORM from ms-swift
```python
# Template from /swift/rewards/orm.py

from swift.rewards import ORM

class MyORM(ORM):
    def __call__(self, completions: List[str], **kwargs) -> torch.Tensor:
        """
        Args:
            completions: Generated model outputs
            **kwargs: Optional context data
        
        Returns:
            torch.Tensor: Rewards of shape (batch_size,)
        """
        rewards = []
        for completion in completions:
            reward = self.compute_score(completion)
            rewards.append(reward)
        return torch.tensor(rewards, dtype=torch.float32)
    
    def compute_score(self, completion: str) -> float:
        # Your scoring logic
        pass
```

### Registering Custom Rewards
```python
# In /swift/rewards/__init__.py

from your_module import YourORM

orms['your_reward_name'] = YourORM

# Then use: --reward_funcs your_reward_name
```

### Training Command
```bash
swift rlhf \
    --rlhf_type grpo \
    --model model_id \
    --dataset dataset_id \
    --reward_funcs reward_func_1 reward_func_2 \
    --reward_weights 0.7 0.3 \
    --num_generations 4 \
    --output_dir output_dir
```

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| vLLM OOM | batch too large | Reduce `per_device_train_batch_size` |
| Reward = 0 always | Bad JSON parsing | Check format of model output |
| Training diverges | LR too high | Use 1e-6 or lower |
| Loss = NaN | Numerical issue | Check reward range [0,1] |
| Slow generation | vLLM config | Use "colocate" mode, tune memory_utilization |

---

## Timeline (Realistic)

| Phase | Task | Time | Dependencies |
|-------|------|------|--------------|
| 1 | Create reward classes | 1 day | None |
| 2 | Unit test rewards | 0.5 day | Phase 1 |
| 3 | Config GRPO | 1 day | Phase 1-2 |
| 4 | Small-scale test | 1-2 days | Phase 3 |
| 5 | Documentation | 1 day | Phase 1-4 |
| **Total** | - | **4-5 days** | - |

---

## Next Step

✅ **READY**: All background research complete
⏳ **NEXT**: Choose reward strategy (recommend: Composite)
🚀 **THEN**: Implement Phase 1 (Reward Functions)

**Decision**: Proceed with Phase 1? (Yes/No/Customize)

---

*Quick reference for GRPO implementation team*
