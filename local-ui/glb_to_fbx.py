import bpy
import sys

argv = sys.argv
argv = argv[argv.index("--") + 1:]
input_path = argv[0]
output_path = argv[1]

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=input_path)
bpy.ops.export_scene.fbx(filepath=output_path, embed_textures=True, path_mode='COPY')
print(f"CONVERT_DONE:{output_path}")
