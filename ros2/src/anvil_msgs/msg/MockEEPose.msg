# Fake-hardware-only extension of CommandedEEPose — published EXCLUSIVELY by
# test/fake_hardware/fake_hardware_node.py on /ee_pose_{arm}, and subscribed
# EXCLUSIVELY by strategies/multi_process.py (when the mock_ee_pose_echo ROS
# param is true) and gt_replay_verifier_node.py. NEVER crosses to real
# hardware — that's the whole point of this being a separate message type
# rather than a field added to CommandedEEPose itself: real hardware's
# /commanded_ee_{arm} and /ee_pose_{arm} contract must stay byte-identical to
# anvil-workcell's copy of CommandedEEPose.msg, with zero fake-hardware-only
# fields riding along on it.
#
# `sequence` is a monotonically-incrementing counter — deliberately NOT
# derived from base.header.stamp. Wall-clock timestamps can be non-monotonic
# (clock corrections, NTP sync, cross-thread/cross-process jitter) and only
# answer "when did this happen," not "is this strictly newer than the last
# message I consumed." This field answers the latter directly, with no
# dependency on clock behavior.
#
# Exists to work around a bug specific to the mock's echo: a pure-software
# echo with no independent physical sensor driving it can, in rare timing
# conditions, re-publish a not-yet-updated pose — indistinguishable from a
# genuinely fresh one by content alone. `sequence` lets a consumer
# (SequenceStalenessGuard) detect this and refuse to compose against it. Real
# hardware's pose is FK-derived from physical joint encoders via an
# independent, continuously-updating loop — it structurally cannot exhibit
# this "hasn't processed the command yet" failure mode, which is exactly why
# this mechanism has no real-hardware counterpart and shouldn't grow one.
#
# Must increment once per underlying pose UPDATE (see fake_hardware_node.py's
# _ee_command_callback), not once per publish tick (publish_ee_poses) — a
# publish tick that re-sends a not-yet-refreshed pose must repeat the same
# sequence value, not advance it, or staleness becomes undetectable.
CommandedEEPose base
uint64 sequence
