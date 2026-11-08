"""Compact xctrace XML to labeled GPU intervals and relevant counter samples."""
import csv
import bisect
import collections
import json
from pathlib import Path
import re
import sys
import subprocess
import xml.etree.ElementTree as ET

out = Path(sys.argv[1])
prefix = sys.argv[2] if len(sys.argv) > 2 else "profile"
output_prefix = "" if prefix == "profile" else prefix + "-"
stream = "--stream-counters" in sys.argv
counter_input = out / (prefix + "-counter-intervals.xml")
process = None
if stream:
    log = (out / (prefix + "-stream-export.stderr")).open("w")
    process = subprocess.Popen([
        "xcrun", "xctrace", "export", "--input", str(out / (prefix + ".trace")),
        "--xpath", '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-counter-intervals"]',
    ], stdout=subprocess.PIPE, stderr=log)
    counter_input = process.stdout


def rows(path):
    references = {}
    for _, element in ET.iterparse(path, events=("end",)):
        if "id" in element.attrib:
            references[element.attrib["id"]] = (element.text, element.attrib.get("fmt", ""))
        if element.tag != "row":
            continue
        values = []
        for child in element:
            if "ref" in child.attrib:
                values.append(references[child.attrib["ref"]])
            else:
                values.append((child.text, child.attrib.get("fmt", "")))
        yield values
        element.clear()


wanted = {"Kernel Occupancy", "ALU Utilization", "F32 Utilization",
          "Instruction Throughput Utilization", "Total SIMD Groups Inflight",
          "Compute SIMD Groups Inflight", "Occupancy Manager Target"}
sample_path = out / (output_prefix + "counter-samples.csv")
with sample_path.open("w") as f:
    writer = csv.writer(f)
    writer.writerow(["start_ns", "duration_ns", "counter", "value"])
    count = 0
    for row in rows(counter_input):
        name = row[3][1]
        if name not in wanted:
            continue
        writer.writerow([row[0][0], row[1][0], name, row[5][0]])
        count += 1
print("counter samples", count)
if process is not None:
    assert process.wait() == 0, "xctrace stream export failed"
    log.close()

pattern = re.compile(r"(fma4|rr_uniform) T(\d+) W(\d+) G(\d+) r(\d+)")
interval_path = out / (output_prefix + "labeled-gpu-intervals.csv")
with interval_path.open("w") as f:
    writer = csv.writer(f)
    writer.writerow(["start_ns", "duration_ns", "kernel", "tile", "width", "groups",
                     "repetition", "channel", "depth", "label", "process"])
    count = 0
    for row in rows(out / (prefix + "-gpu-intervals.xml")):
        label = row[6][1]
        match = pattern.search(label)
        if match:
            writer.writerow([row[0][0], row[1][0], *match.groups(), row[2][1],
                             row[5][0], label, row[10][1]])
            count += 1
print("labeled GPU intervals", count)

# Correlate samples entirely inside the middle 80% of a labeled GPU interval.
# Exclude warmups and weight by the counter's sample interval duration.
samples = collections.defaultdict(list)
with sample_path.open() as f:
    for row in csv.DictReader(f):
        samples[row["counter"]].append((int(row["start_ns"]), int(row["duration_ns"]), float(row["value"])))
starts = {key: [r[0] for r in values] for key, values in samples.items()}
aggregate = collections.defaultdict(lambda: [0., 0., 0., 0])
with interval_path.open() as f:
    for row in csv.DictReader(f):
        if int(row["repetition"]) == 0:
            continue
        start = int(row["start_ns"])
        duration = int(row["duration_ns"])
        end = start + duration - int(duration * .1)
        start += int(duration * .1)
        for name, values in samples.items():
            index = bisect.bisect_left(starts[name], start)
            while index < len(values) and values[index][0] < end:
                sample_start, sample_duration, value = values[index]
                if sample_start + sample_duration <= end:
                    key = (row["kernel"], int(row["tile"]), int(row["width"]), int(row["groups"]), name)
                    acc = aggregate[key]
                    acc[0] += value * sample_duration
                    acc[1] += sample_duration
                    acc[2] = max(acc[2], value)
                    acc[3] += 1
                index += 1
result = []
for key, (total, duration, maximum, count) in sorted(aggregate.items()):
    row = dict(zip(["kernel", "tile", "width", "groups", "counter"], key))
    row.update(weighted_mean=total/duration, max_sample=maximum, samples=count, sample_duration_ns=duration)
    result.append(row)
(out / (output_prefix + "counter-summary.json")).write_text(json.dumps(result, indent=2) + "\n")
print("counter summary rows", len(result))
