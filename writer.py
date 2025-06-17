import mujoco

xml_path = '/home/jzq/github/dex-serl/franka_sim/franka_sim/envs/xmls/dpfrankacube/panda_dex.xml'
# xml_path = '/home/jzq/github/dex-serl/franka_sim/franka_sim/envs/xmls/dpfrankacube/DPhand/dphand_arena.xml'

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)


# import xml.etree.ElementTree as ET

# tree = ET.parse(xml_path)  # 替换为你的模型文件
# root = tree.getroot()

# joint_names = []
# for joint in root.findall('.//joint'):
#     joint_names.append(joint.get('name'))
# print(f"找到 {len(joint_names)} 个关节: {joint_names}")

# sensors = root.find('sensor')
# if sensors is None:
#     sensors = ET.SubElement(root, 'sensor')  # 创建新的 <sensor> 节点

# for name in joint_names:
#     if name != None:
#         ET.SubElement(sensors, 'jointpos', name=f"{name}_pos", joint=name)
        

# tree.write('robot_with_sensors.xml', encoding='utf-8', xml_declaration=True)


import xml.etree.ElementTree as ET

# 读取原始 XML
tree = ET.parse(xml_path)
root = tree.getroot()

# 获取或创建 <sensor> 标签
sensors = root.find('sensor')
if sensors is None:
    sensors = ET.SubElement(root, 'sensor')

# 为每个关节添加 jointpos 传感器，并手动换行+缩进
for joint in root.findall('.//joint'):
    joint_name = joint.get('name')

    if joint_name != None:
        sensor = ET.SubElement(sensors, 'jointpos', name=f"{joint_name}_pos", joint=joint_name)
        sensor.tail = '\n  '  # 换行 + 2空格缩进（</sensor> 对齐）

# 最后一个元素换行对齐 </sensor>
if len(sensors) > 0:
    sensors[-1].tail = '\n'

# 保存（启用缩进格式化）
ET.indent(tree, space='  ')  # Python 3.9+
tree.write('robot_with_sensors_pretty.xml', encoding='utf-8')
