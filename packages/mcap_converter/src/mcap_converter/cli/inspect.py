#!/usr/bin/env python3
"""
MCAP File Structure Analysis Tool

This script analyzes MCAP file data structure and types, including:
- All available topics
- Message type for each topic
- Message count for each topic
- Message data structure and field types
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from collections import defaultdict
import json
from datetime import datetime


def normalize_timestamp(value: Union[int, float, datetime, None]) -> Optional[float]:
    """
    Convert MCAP log_time to seconds (float). MCAP log_time usually nanosecond integer,
    but could also be datetime or other types.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.timestamp()
    if isinstance(value, (int, float)):
        # Most MCAP uses nanoseconds; if value is large, convert to seconds
        return float(value) / 1e9 if abs(value) > 1e6 else float(value)
    return None

try:
    from mcap_ros2.reader import read_ros2_messages
    from mcap.reader import make_reader
except ImportError as e:
    print(f"Error: Missing required dependencies. Please install:")
    print(f"  pip install mcap mcap-ros2-support")
    print(f"Detailed error: {e}")
    sys.exit(1)

def get_topic_info(mcap_path: str) -> Dict[str, Any]:
    """
    Get basic information of all topics in MCAP file

    Returns:
        {
            'topic_name': {
                'message_type': str,
                'message_count': int,
                'first_timestamp': float,
                'last_timestamp': float,
            }
        }
    """
    topic_info = {}

    print(f"Reading MCAP file: {mcap_path}")

    with open(mcap_path, "rb") as f:
        reader = make_reader(f)

        # Prioritize using summary (if exists) to get channel/schema information
        summary = None
        try:
            summary = reader.get_summary()
        except Exception:
            summary = None

        if summary:
            for channel in summary.channels.values():
                topic_name = channel.topic
                schema = summary.schemas.get(channel.schema_id) if channel.schema_id else None
                if schema and schema.name:
                    message_type = schema.name.replace("ros2msg://", "")
                    schema_name = schema.name
                else:
                    message_type = getattr(channel, "message_encoding", "unknown")
                    schema_name = schema.name if schema else 'unknown'

                topic_info[topic_name] = {
                    'message_type': message_type,
                    'schema_name': schema_name,
                    'message_count': 0,
                    'first_timestamp': None,
                    'last_timestamp': None,
                }

        # through iter_messages count message data (and fill in summary non-existent topics)
        for schema, channel, message in reader.iter_messages():
            topic_name = channel.topic
            if topic_name not in topic_info:
                if schema and schema.name:
                    message_type = schema.name.replace("ros2msg://", "")
                    schema_name = schema.name
                else:
                    message_type = getattr(channel, "message_encoding", "unknown")
                    schema_name = schema.name if schema else 'unknown'
                topic_info[topic_name] = {
                    'message_type': message_type,
                    'schema_name': schema_name,
                    'message_count': 0,
                    'first_timestamp': None,
                    'last_timestamp': None,
                }

            info = topic_info[topic_name]
            info['message_count'] += 1
            timestamp = normalize_timestamp(message.log_time)
            if timestamp is not None:
                if info['first_timestamp'] is None:
                    info['first_timestamp'] = timestamp
                info['last_timestamp'] = timestamp

    return topic_info


def inspect_message_structure(mcap_path: str, topic: str = None, max_samples: int = 5) -> Dict[str, Any]:
    """
    Inspect message structure of specified topic

    Args:
        mcap_path: MCAP File path
        topic: Topic to inspect (if None, inspect all topics)
        max_samples: Maximum number of messages to inspect per topic

    Returns:
        Dictionary containing message structure information
    """
    structure_info = {}
    topic_samples = defaultdict(list)

    print(f"Analyzing message structure...")

    # Read messages and collect samples
    topics_to_inspect = [topic] if topic else None

    try:
        for idx, message in enumerate(read_ros2_messages(mcap_path, topics=topics_to_inspect)):
            topic_name = message.channel.topic

            if topic and topic_name != topic:
                continue

            if len(topic_samples[topic_name]) < max_samples:
                ros_msg = message.ros_msg

                # Extract message structure
                msg_dict = {}
                extract_message_fields(ros_msg, msg_dict, prefix="")

                topic_samples[topic_name].append({
                    'timestamp': normalize_timestamp(message.log_time),
                    'structure': msg_dict,
                })
    except Exception as e:
        print(f"Warning: Error reading ROS2 messages: {e}")
        print("Trying to use basic MCAP reader...")
        return {}

    # Analyze structure
    for topic_name, samples in topic_samples.items():
        if not samples:
            continue

        # Merge fields from all samples
        all_fields = {}
        for sample in samples:
            merge_structure(all_fields, sample['structure'])

        structure_info[topic_name] = {
            'samples_analyzed': len(samples),
            'fields': all_fields,
            'first_timestamp': samples[0]['timestamp'],
            'last_timestamp': samples[-1]['timestamp'],
        }

    return structure_info


def extract_message_fields(obj: Any, result: Dict, prefix: str = ""):
    """
    Recursively extract fields and types from ROS2 messages
    """
    if hasattr(obj, '__slots__'):
        # ROS2 message object
        for slot in obj.__slots__:
            field_name = f"{prefix}.{slot}" if prefix else slot
            value = getattr(obj, slot, None)

            if value is None:
                result[field_name] = {'type': 'None', 'value': None}
            elif isinstance(value, (int, float, str, bool)):
                result[field_name] = {
                    'type': type(value).__name__,
                    'value': value,
                }
            elif isinstance(value, (list, tuple)):
                if len(value) > 0:
                    # Check list element type
                    elem_type = type(value[0]).__name__
                    result[field_name] = {
                        'type': f'List[{elem_type}]',
                        'length': len(value),
                        'sample_value': value[0] if len(value) > 0 else None,
                    }
                    # If complex object, recursively extract first element
                    if hasattr(value[0], '__slots__'):
                        extract_message_fields(value[0], result, f"{field_name}[0]")
                else:
                    result[field_name] = {'type': 'List[empty]', 'length': 0}
            elif hasattr(value, '__slots__'):
                # Nested ROS2 message
                result[field_name] = {'type': 'ROS2Message', 'fields': {}}
                extract_message_fields(value, result, field_name)
            else:
                result[field_name] = {'type': type(value).__name__, 'value': str(value)[:100]}
    elif isinstance(obj, dict):
        for key, value in obj.items():
            field_name = f"{prefix}.{key}" if prefix else key
            result[field_name] = {'type': type(value).__name__, 'value': value}
    else:
        result[prefix or 'root'] = {'type': type(obj).__name__, 'value': str(obj)[:100]}


def merge_structure(target: Dict, source: Dict):
    """
    Merge two structure dictionaries, preserving type information
    """
    for key, value in source.items():
        if key not in target:
            target[key] = value.copy()
        else:
            # If types differ, mark as variable type
            if target[key].get('type') != value.get('type'):
                target[key]['type'] = f"{target[key].get('type')} | {value.get('type')}"


def format_output(topic_info: Dict, structure_info: Dict, output_format: str = "text"):
    """
    Format output results
    """
    if output_format == "json":
        return json.dumps({
            'topics': topic_info,
            'structures': structure_info,
        }, indent=2, default=str)

    # Text format output
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("MCAP File Structure Analysis Results")
    output_lines.append("=" * 80)
    output_lines.append("")

    # Topics Summary
    output_lines.append("Topics Summary")
    output_lines.append("-" * 80)
    output_lines.append(f"{'Topic':<50} {'Type':<30} {'Msg Count':<10}")
    output_lines.append("-" * 80)

    for topic_name, info in sorted(topic_info.items()):
        msg_type = info['message_type']
        count = info['message_count']
        output_lines.append(f"{topic_name:<50} {msg_type:<30} {count:<10}")

    output_lines.append("")

    # Detailed Structure
    if structure_info:
        output_lines.append("Message Structure Details")
        output_lines.append("-" * 80)

        for topic_name, struct_info in sorted(structure_info.items()):
            output_lines.append(f"\nTopic: {topic_name}")
            output_lines.append(f"  Samples analyzed: {struct_info['samples_analyzed']}")
            output_lines.append(f"  Time range: {struct_info['first_timestamp']:.6f} - {struct_info['last_timestamp']:.6f} s")
            output_lines.append("  Field structure:")

            for field_name, field_info in sorted(struct_info['fields'].items()):
                field_type = field_info.get('type', 'unknown')
                if 'length' in field_info:
                    output_lines.append(f"    {field_name:<60} {field_type} (length: {field_info['length']})")
                elif 'value' in field_info and field_info['value'] is not None:
                    value_str = str(field_info['value'])[:50]
                    output_lines.append(f"    {field_name:<60} {field_type} = {value_str}")
                else:
                    output_lines.append(f"    {field_name:<60} {field_type}")

    return "\n".join(output_lines)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze MCAP file data structure and types",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze entire MCAP file
  mcap-inspect path/to/file.mcap

  # Only analyze specific topic
  mcap-inspect path/to/file.mcap --topic /leader1/joint_states

  # Output as JSON format
  mcap-inspect path/to/file.mcap --format json

  # Increase sample count for more detailed analysis
  mcap-inspect path/to/file.mcap --max-samples 10
        """
    )

    parser.add_argument(
        "mcap_path",
        type=str,
        help="MCAP File path"
    )

    parser.add_argument(
        "--topic",
        type=str,
        default=None,
        help="Only analyze specified topic (if not specified, analyze all topics)"
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=5,
        help="Maximum message samples to analyze per topic (default: 5)"
    )

    parser.add_argument(
        "--format",
        type=str,
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (if not specified, output to stdout)"
    )

    args = parser.parse_args()

    # Check if file exists
    mcap_path = Path(args.mcap_path)
    if not mcap_path.exists():
        print(f"Error: File does not exist: {mcap_path}")
        sys.exit(1)

    if not mcap_path.is_file():
        print(f"Error: Not a file: {mcap_path}")
        sys.exit(1)

    try:
        # Get topics information
        print("Reading topics information...")
        topic_info = get_topic_info(str(mcap_path))

        if not topic_info:
            print("Warning: No topics found")
            sys.exit(1)

        # Analyze message structure
        print("Analyzing message structure...")
        structure_info = inspect_message_structure(
            str(mcap_path),
            topic=args.topic,
            max_samples=args.max_samples
        )

        # Format output
        output = format_output(topic_info, structure_info, args.format)

        # Output results
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(output)
            print(f"\n[OK] Results saved to: {output_path}")
        else:
            print(output)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
