from typing import Tuple

print("Reading...")
file = open("VitalTriAM_Optimized.rou.xml")

lines = []

flow_id = 0
ignore = 0
for i, line in enumerate(file.readlines()):
    if line.find("<flow") == -1:
        flow_id += 1
    else:
        id_begin = line.find('id="') + 4
        id_end = line.find('" begin')
        probability = float(line[line.find('probability="') + 13: line.find('">')])
        if probability == 1.0:
            ignore = 1
        flow_id = 10 * int(line[id_begin:id_end]) + 10
    if line.find("</route") != -1:
        flow_id = 99999999  # bro don't ask
        # lines.append((flow_id, line))
        # continue
    if not ignore:
        lines.append((flow_id, line))
    elif ignore == 3:
        ignore = 0
    else:
        ignore += 1
file.close()

print("Sorting and Writing...")
file = open("VitalTriAM_Optimized_Sorted.rou.xml", "w")
lines.sort(key=lambda id_line: id_line[0])
# for flow_id, line in sorted(lines, key=lambda id_line: id_line[0]):
for flow_id, line in lines:
    file.write(line)
file.close()
print("Done!")