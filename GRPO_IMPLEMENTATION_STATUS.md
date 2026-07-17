# GRPO Implementation for Camera Movement VQA - Status Report

**Date**: 2026-05-25  
**Status**: ✅ READY FOR IMPLEMENTATION  
**Phase**: Plan Review Complete, Awaiting Execution Decision

---

## Executive Summary

Background research is complete. Two specialized agents have thoroughly explored:
1. **ms-swift GRPO Infrastructure** (4.3.0) - 100% mapped
2. **Camera Movement Evaluation Logic** - Fully documented with 7 reward function templates

The system is now **ready for Phase 1 implementation** (Reward Functions).

---

## What We Know

### 1. ms-swift GRPO Capabilities ✓

**How to Launch GRPO Training**:
```bash
swift rlhf \
  --rlhf_type grpo \
  --model model_id \
  --dataset dataset_id \
  --reward_funcs accuracy format \
  --reward_weights 0.8 0.2 \
  --num_generations 8 \
  --output_dir ./output
```

**Key Components**:
- `GRPOTrainer`: Main training class in `/swift/rlhf_trainers/grpo_trainer.py` (2632 lines)
- `GRPOConfig`: Configuration in `/swift/rlhf_trainers/arguments.py`
- Built-in rewards: `MathAccuracy`, `Format`, `ReActFormat`, `CosineReward`, `RepetitionPenalty`, `SoftOverlong`, `ReactORM`, `MathORM`
- Plugins: `DefaultRMPlugin` (discriminative), `GenRMPlugin` (LLM-as-judge)
- Generation: vLLM integration with "colocate" or "server" modes
- Requirements: `num_generations >= 2`

**Real Example**:
- Location: `/examples/train/grpo/external/grpo_7b.sh`
- Setup: 6 GPUs, Qwen2.5-7B model
- Demonstrates full pipeline with custom reward functions

### 2. Camera Movement Evaluation Logic ✓

**The 15 Movement Types**:

| Type | Direction | Category |
|------|-----------|----------|
| Pan, Tilt, Truck, Crane, Arc, Roll | Required | Rotational/Translational |
| Dolly In, Dolly Out, Zoom In, Zoom Out | None | Physical/Optical |
| Follow, Free Fly, Static, Unstable | None | Complex/Static |
| Focus Shift | None | Special |

**F1 Score Computation**:
- **Frame-level**: Multi-label set comparison at 0.1s intervals
- **Composite labels**: Type + Direction (e.g., "Pan_left", "Crane_up")
- **Partial credit**: Each label scored independently
- **Segment-level**: IoU-based temporal grounding (threshold=0.5)

Example: If GT has `{Pan_left, Dolly_In}` and Pred has `{Pan_left}`:
- TP=1, FP=0, FN=1 → F1 = 0.67 (partial credit, not 0)

**JSON Format** (Model Output):
```json
{
  "video_id": "video_123",
  "segments": [
    {
      "start_time": 0.0,
      "end_time": 2.5,
      "basic_movement": [
        {"type": "Pan", "direction": "left", "speed": "medium"}
      ],
      "special_movement": ["Handheld"],
      "confidence": "high",
      "visual_evidence": "Background pans uniformly left"
    }
  ]
}
```

### 3. Data Reuse Strategy ✓

**SFT→RL Transition** (Recommended):
- **Phase 1 (SFT)**: Train on full dataset with reference solutions
- **Phase 2 (RL)**: 
  - Reuse same video IDs and questions
  - Remove reference solutions (use only as GT for evaluation)
  - GRPO generates new outputs during training
  - Model learns to improve upon SFT baseline

**Format Preservation**:
- Format constraints can be enforced via reward function
- Recommend: 10-20% weight for format reward
- Won't lose JSON/choice capability if included

---

## 7 Reward Function Options (Templated)

| # | Name | Complexity | Signal | When to Use |
|----|------|-----------|--------|------------|
| 1 | Basic F1 | Simple | Frame-level F1 only | MVP, fast iteration |
| 2 | Per-Label | Medium | Individual label F1s | Debug/analysis |
| 3 | Hierarchical | Medium | Type (0.7) + Direction (0.3) | Direction sensitivity |
| 4 | Segment IoU | Medium | Temporal grounding | Fix timing issues |
| 5 | **Composite** ⭐ | Complex | Frame F1 + Segment + Special + Speed | Production (RECOMMENDED) |
| 6 | Confidence-Weighted | Medium | Penalize high-confidence errors | Calibration |
| 7 | Class-Balanced | Medium | Weight rare movements higher | Long-tail coverage |

**Recommendation**: Start with **Option 5 (Composite)** for balanced training signal.

---

## Implementation Roadmap (4-5 Days)

### Phase 1: Reward Functions [1-2 days] ⚡ CRITICAL
**Create**:
- `/camera_movement_sft/rl/rewards.py` - Core ORM classes
- `/camera_movement_sft/rl/unit_tests_reward.py` - Unit tests

**Implement**:
1. `CameraMovementORM` base class (extends ms-swift ORM)
2. 3-4 reward function variants (Basic F1, Composite, Format)
3. Integration with existing evaluation code
4. Verification tests

### Phase 2: Training Config [1-2 days]
**Create**:
- `/camera_movement_sft/rl/config_grpo.yaml` - Hyperparameters
- `/camera_movement_sft/rl/train_grpo.py` - Training script (optional)

**Configure**:
- Data loading (reuse SFT dataset)
- Model and generation setup
- Hyperparameters (learning_rate=1e-6, beta=0.04, etc.)

### Phase 3: Documentation [1 day]
- `GRPO_IMPLEMENTATION_GUIDE.md` - Complete architecture
- `CAMERA_MOVEMENT_REWARD_DESIGN.md` - Detailed reward design
- `SFT_TO_RL_BEST_PRACTICES.md` - Pipeline recommendations

### Phase 4: Validation & Examples [1-2 days]
- Small-scale test run (100-500 examples, 1-2 GPUs)
- Before/after metrics comparison
- Example configs and runnable scripts

---

## Critical Files (Ready to Reference)

### From Background Research:
- **Template Rewards**: `/tmp/camera_movement_analysis.md` (comprehensive guide)
- **GRPO Examples**: `/examples/train/grpo/external/grpo_7b.sh`
- **ms-swift ORM Classes**: `/swift/rewards/orm.py` (base classes + 8 examples)

### To Create:
- `/camera_movement_sft/rl/rewards.py` - Custom reward functions
- `/camera_movement_sft/rl/config_grpo.yaml` - Training config
- `/camera_movement_sft/rl/unit_tests_reward.py` - Tests

---

## User Decisions Needed

### 1. Reward Function Strategy
- [ ] Start simple (Basic F1)?
- [ ] Go composite (multi-signal)?
- **Recommendation**: Composite (balanced, proven approach)

### 2. Training Resources
- [ ] Small (1-2 GPU, 500 examples)?
- [ ] Medium (4-8 GPU, 5K examples)?
- [ ] Large (16+ GPU, 50K+)?
- **Impacts**: Batch size, learning rate, num_generations

### 3. Format Constraint Priority
- [ ] High (30% of reward)?
- [ ] Medium (10-20%)?
- [ ] Low (5%)?
- **Recommendation**: Medium (10-20%) for baseline

### 4. Timeline
- [ ] MVP only (2-3 days)?
- [ ] Full implementation (5+ days)?
- **Recommendation**: Full (ensures production-ready)

### 5. Proceed or Wait?
- [ ] **PROCEED NOW** with defaults (recommended)
- [ ] Get user input first
- [ ] Something else?

---

## Recommended Next Steps

### Option A: START IMMEDIATELY ⚡ (Recommended)
```bash
# Use sensible defaults:
# - Reward: Composite (multi-signal)
# - Resources: Medium (4 GPU, 5K examples)
# - Format weight: 15%
# - Timeline: Full implementation

# User can provide feedback in parallel
# Phase 1 (Rewards) can start today
```

### Option B: Get User Input First
- Answer the 5 decision points above
- Customize defaults
- Then proceed

---

## Key Metrics to Track

| Metric | Baseline | Target | How |
|--------|----------|--------|-----|
| Composite F1 | SFT | +5-10% | Main reward signal |
| Format Compliance | ~100% | ~100% | Include format reward |
| Inference Speed | Baseline | <5% slower | Monitor during training |
| Model Diversity | Low (SFT) | Higher | GRPO generates multiple outputs |

---

## Risk Mitigation

✅ **Strategies**:
1. Start with 100-500 examples (quick iteration cycle)
2. Use conservative learning rate (1e-6) to preserve SFT capabilities
3. Include format constraint in reward (maintain JSON/choice output)
4. Regular checkpointing and validation monitoring
5. Compare metrics before/after each phase

---

## Resources & References

### Code:
- ms-swift GRPO: `/swift/rlhf_trainers/grpo_trainer.py`
- ORM base classes: `/swift/rewards/orm.py`
- Examples: `/examples/train/grpo/external/`

### Generated Docs:
- Camera evaluation guide: `/tmp/camera_movement_analysis.md`
- Plan details: `/root/.claude-internal/plans/replicated-crafting-shamir-agent-a0378a146e579b9fb.md`

### Papers/References:
- GRPO: https://arxiv.org/abs/2402.05191
- vLLM: https://arxiv.org/abs/2309.06180

---

## Status Timeline

| Date | What | Status |
|------|------|--------|
| 2026-05-25 | Background research (2 agents) | ✅ Complete |
| 2026-05-25 | Plan review & documentation | ✅ Complete |
| Today | **Phase 1 Ready** | ⏳ Awaiting decision |
| +1-2 days | Phase 1: Reward functions | ⏳ Pending |
| +2-4 days | Phase 2-3: Config & Docs | ⏳ Pending |
| +4-5 days | Phase 4: Validation | ⏳ Pending |

---

## Next Action

**Ready to proceed with Phase 1 implementation?** 

Options:
1. ✅ **Yes, start immediately** (use defaults)
2. ⚠️ **Yes, but customize first** (answer 5 questions)
3. 📋 **Need more information** (specify what)

---

*Generated by GRPO Implementation Planning Team*  
*Background Agents: a0378a146e579b9fb (GRPO), a78d33eb7136a6cf5 (Evaluation)*
