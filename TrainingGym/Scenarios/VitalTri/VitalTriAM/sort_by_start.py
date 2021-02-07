from typing import Tuple

print("Reading...")
file = open("VitalTriAM_Optimized_Sorted.rou.xml")

lines = []

flow_id = 0
for i, line in enumerate(file.readlines()):
    if line.find("<flow") == -1:
        flow_id += 1
    else:
        begin = float(line[line.find('begin="') + 7: line.find('" end=')])
        flow_id = begin * 100000 + i * 10
    if line.find("</route") != -1:
        flow_id = 9999999999999  # bro don't ask
    lines.append((flow_id, line))
file.close()

print("Sorting and Writing...")
file = open("VitalTriAM_Optimized_New.rou.xml", "w")
lines.sort(key=lambda id_line: id_line[0])
# for flow_id, line in sorted(lines, key=lambda id_line: id_line[0]):
for flow_id, line in lines:
    file.write(line)
file.close()
print("Done!")