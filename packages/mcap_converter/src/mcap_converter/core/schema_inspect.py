"""Deep per-message field-structure inspection for a single MCAP topic.

Ported from the old standalone `mcap-inspect` CLI (now folded into `mcap-valid --topic`).
Unlike core/quality.py's summary-first approach (fast, O(1), footer-only), this module does
a full read_ros2_messages() scan to sample actual message field values/types — a genuinely
different, heavier operation, kept in its own module for that reason.
"""

from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from mcap_ros2.reader import read_ros2_messages


def normalize_timestamp(value: Union[int, float, "datetime", None]) -> Optional[float]:
    """Convert an MCAP log_time to seconds (float). Usually a nanosecond int, but may be a datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.timestamp()
    if isinstance(value, (int, float)):
        return float(value) / 1e9 if abs(value) > 1e6 else float(value)
    return None


def extract_message_fields(obj: Any, result: Dict, prefix: str = "") -> None:
    """Recursively extract field names/types/sample-values from a ROS2 message object."""
    if hasattr(obj, "__slots__"):
        # ROS2 message object
        for slot in obj.__slots__:
            field_name = f"{prefix}.{slot}" if prefix else slot
            value = getattr(obj, slot, None)

            if value is None:
                result[field_name] = {"type": "None", "value": None}
            elif isinstance(value, (int, float, str, bool)):
                result[field_name] = {
                    "type": type(value).__name__,
                    "value": value,
                }
            elif isinstance(value, (list, tuple)):
                if len(value) > 0:
                    # Check list element type
                    elem_type = type(value[0]).__name__
                    result[field_name] = {
                        "type": f"List[{elem_type}]",
                        "length": len(value),
                        "sample_value": value[0] if len(value) > 0 else None,
                    }
                    # If complex object, recursively extract first element
                    if hasattr(value[0], "__slots__"):
                        extract_message_fields(value[0], result, f"{field_name}[0]")
                else:
                    result[field_name] = {"type": "List[empty]", "length": 0}
            elif hasattr(value, "__slots__"):
                # Nested ROS2 message
                result[field_name] = {"type": "ROS2Message", "fields": {}}
                extract_message_fields(value, result, field_name)
            else:
                result[field_name] = {"type": type(value).__name__, "value": str(value)[:100]}
    elif isinstance(obj, dict):
        for key, value in obj.items():
            field_name = f"{prefix}.{key}" if prefix else key
            result[field_name] = {"type": type(value).__name__, "value": value}
    else:
        result[prefix or "root"] = {"type": type(obj).__name__, "value": str(obj)[:100]}


def merge_structure(target: Dict, source: Dict) -> None:
    """Merge two structure dicts, marking a field as a union type if samples disagree."""
    for key, value in source.items():
        if key not in target:
            target[key] = value.copy()
        else:
            # If types differ, mark as variable type
            if target[key].get("type") != value.get("type"):
                target[key]["type"] = f"{target[key].get('type')} | {value.get('type')}"


def inspect_message_structure(
    mcap_path: str, topic: Optional[str] = None, max_samples: int = 5
) -> Dict[str, Any]:
    """
    Sample up to `max_samples` messages per topic (or just `topic` if given) and return
    {topic_name: {"samples_analyzed": int, "fields": {...}, "first_timestamp": float, "last_timestamp": float}}.

    Any I/O/parse error from read_ros2_messages() propagates to the caller rather than being
    swallowed — the old standalone `mcap-inspect` tool printed a warning and returned {} on
    ANY exception (including a genuinely unreadable file), which hid real problems. There is
    no per-message defensive handling either: extract_message_fields() only inspects plain
    ROS2 message attributes/lists/scalars, so a mid-stream failure there would indicate a
    genuine bug worth surfacing, not a recoverable per-sample condition.
    """
    structure_info: Dict[str, Any] = {}
    topic_samples: Dict[str, list] = defaultdict(list)

    topics_to_inspect = [topic] if topic else None

    for message in read_ros2_messages(mcap_path, topics=topics_to_inspect):
        topic_name = message.channel.topic

        if topic and topic_name != topic:
            continue

        if len(topic_samples[topic_name]) < max_samples:
            ros_msg = message.ros_msg

            msg_dict: Dict[str, Any] = {}
            extract_message_fields(ros_msg, msg_dict, prefix="")

            topic_samples[topic_name].append(
                {
                    "timestamp": normalize_timestamp(message.log_time),
                    "structure": msg_dict,
                }
            )

    for topic_name, samples in topic_samples.items():
        if not samples:
            continue

        # Merge fields from all samples
        all_fields: Dict[str, Any] = {}
        for sample in samples:
            merge_structure(all_fields, sample["structure"])

        structure_info[topic_name] = {
            "samples_analyzed": len(samples),
            "fields": all_fields,
            "first_timestamp": samples[0]["timestamp"],
            "last_timestamp": samples[-1]["timestamp"],
        }

    return structure_info


def render_structure_text(structure_info: Dict[str, Any]) -> str:
    """
    Render the per-topic field structure as human-readable text (one topic per block).

    Extracted from the old `mcap-inspect` CLI's format_output()'s "Message Structure
    Details" section, minus the "Topics Summary" table (that data now lives in
    mcap-valid's own unified topic-severity table, which already shows topic/type/count
    for every topic).
    """
    if not structure_info:
        return ""

    lines: List[str] = ["Message Structure Details", "-" * 80]

    for topic_name, struct_info in sorted(structure_info.items()):
        lines.append(f"\nTopic: {topic_name}")
        lines.append(f"  Samples analyzed: {struct_info['samples_analyzed']}")
        first_ts = struct_info["first_timestamp"]
        last_ts = struct_info["last_timestamp"]
        if first_ts is not None and last_ts is not None:
            lines.append(f"  Time range: {first_ts:.6f} - {last_ts:.6f} s")
        lines.append("  Field structure:")

        for field_name, field_info in sorted(struct_info["fields"].items()):
            field_type = field_info.get("type", "unknown")
            if "length" in field_info:
                lines.append(f"    {field_name:<60} {field_type} (length: {field_info['length']})")
            elif "value" in field_info and field_info["value"] is not None:
                value_str = str(field_info["value"])[:50]
                lines.append(f"    {field_name:<60} {field_type} = {value_str}")
            else:
                lines.append(f"    {field_name:<60} {field_type}")

    return "\n".join(lines)
