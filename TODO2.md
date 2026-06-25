# TODO2 Note - Minimal Patch Plan for HUG-style Upgrade

Purpose
- Record the smallest safe code edits to unblock Stage II physics learning and checkpoint forward compatibility.
- Keep old checkpoints usable.

Scope
- scripts/sonic_hand_midtrain_sim.py
- scripts/sonic_hand_flow.py

--------------------------------------------------
Patch A: Make Stage II physics loss actually train the model
--------------------------------------------------

Problem
- In midtrain, q_pred is generated inside no_grad, so analytical physics loss cannot backprop to model.

Current pattern
- with torch.no_grad():
    q_pred = flow_sample(model, q_manus, q_robot, n_steps=5, device=device)
- phys = analytical_loss(q_pred)

Replace with
- q_pred = flow_sample(model, q_manus, q_robot, n_steps=5, device=device)
- phys = analytical_loss(q_pred)

Reason
- analytical_loss is differentiable; it must see a tensor attached to graph.

Safety note
- Keep simulation rollout path detached if needed:
  q_np = q_pred[:n_sim].detach().cpu().numpy()

Expected outcome
- coupling/velocity/range terms contribute gradients.
- Stage II really adapts trainable layers to robot constraints.

--------------------------------------------------
Patch B: Add checkpoint schema version and compatibility load path
--------------------------------------------------

B1. Save schema metadata in Stage I checkpoints
File: scripts/sonic_hand_flow.py
Where: train(), checkpoint dictionary

Add fields
- schema_version: 2
- model_version: handflow_v2
- input_spec:
  - q_manus_dim: 20
  - q_robot_dim: 20
  - q_target_dim: 20
  - chunk_size: args.chunk

Example shape
- ckpt = {
    "schema_version": 2,
    "model_version": "handflow_v2",
    "input_spec": {
      "q_manus_dim": N_JOINTS,
      "q_robot_dim": N_JOINTS,
      "q_target_dim": N_JOINTS,
      "chunk_size": args.chunk,
    },
    "step": step,
    "val_loss": val_loss,
    "model": model.state_dict(),
    "opt": opt.state_dict(),
    "args": vars(args),
  }

B2. Robust load in Stage I eval and benchmark
File: scripts/sonic_hand_flow.py
Where: evaluate(), benchmark()

Load policy
- schema_version = ckpt.get("schema_version", 1)
- model.load_state_dict(ckpt["model"], strict=False)
- Print warning if schema_version == 1:
  "legacy checkpoint loaded in compatibility mode"

B3. Robust load and save in Stage II midtrain
File: scripts/sonic_hand_midtrain_sim.py
Where: pretrain load + checkpoint save

Load policy
- schema_version = ckpt.get("schema_version", 1)
- model.load_state_dict(ckpt["model"], strict=False)
- Print compatibility message for v1.

Save policy
- include same schema fields as Stage I
- keep args, model, opt, step, val_loss unchanged

Expected outcome
- Old best.pt still loads.
- New checkpoints self-describe tensor contract.

--------------------------------------------------
Optional micro patch (still minimal)
--------------------------------------------------

Patch C: Jerk smoothness term in Stage II analytical loss
File: scripts/sonic_hand_midtrain_sim.py

Add
- ddq = q[:, 2:, :] - 2 * q[:, 1:-1, :] + q[:, :-2, :]
- losses["jerk"] = w_jerk * ddq.pow(2).mean()

CLI arg
- --w_jerk default 0.1

Benefit
- Reduces high-frequency chatter in generated trajectories.

--------------------------------------------------
Quick test checklist
--------------------------------------------------

1) Backprop sanity
- Run 200 to 500 steps Stage II analytical mode.
- Confirm coupling/velocity/range decrease and train loss changes.

2) Legacy checkpoint load
- Use old checkpoints/sonic_hand/best.pt to start Stage II.
- Verify no shape crash and compatibility warning appears.

3) New checkpoint reload
- Train briefly, save new best.pt, reload with eval/benchmark.
- Verify no error and schema_version printed as 2.

--------------------------------------------------
Command examples
--------------------------------------------------

Stage II analytical only
conda run -n env_isaaclab python scripts/sonic_hand_midtrain_sim.py \
  --pretrain checkpoints/sonic_hand/best.pt \
  --ckpt_dir checkpoints/sonic_hand_sim \
  --steps 20000 --batch 1024

Stage II with IsaacLab
conda run -n env_isaaclab python scripts/sonic_hand_midtrain_sim.py \
  --pretrain checkpoints/sonic_hand/best.pt \
  --ckpt_dir checkpoints/sonic_hand_sim \
  --steps 20000 --use_isaaclab --sim_envs 32
