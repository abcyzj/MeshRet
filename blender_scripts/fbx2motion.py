import sys
from argparse import ArgumentParser

import bpy
import numpy as np


def delete_hierarchy(obj):
    names = set([obj.name])

    # recursion
    def get_child_names(obj):
        for child in obj.children:
            names.add(child.name)
            if child.children:
                get_child_names(child)

    get_child_names(obj)

    objects = bpy.data.objects
    for n in names:
        objects[n].select_set(True)

    bpy.ops.object.delete()


def main(args):
    bpy.ops.import_scene.fbx(filepath=args.input_fbx, automatic_bone_orientation=True)
    if bpy.context.object.animation_data is None:
        sys.exit(10)
    a = bpy.context.object.animation_data.action
    frame_start, frame_end = int(a.frame_range[0]), int(a.frame_range[1])
    scene = bpy.context.scene
    arm = None
    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE':
            arm = obj
            break
    if arm is None:
        sys.exit(11)
    bone_names = [b.name for b in arm.data.bones]
    bone_rest_mats = np.stack([bone.matrix_local for bone in arm.data.bones], axis=0)
    bone_pose_mats = np.zeros([frame_end - frame_start + 1, len(bone_names), 4, 4])
    frametime = 1.0 / (scene.render.fps / scene.render.fps_base)
    for frame in range(frame_start, frame_end + 1):
        scene.frame_set(frame)
        for name in bone_names:
            pose_bone = arm.pose.bones[name]
            bone_pose_mats[frame - frame_start, bone_names.index(name)] = pose_bone.matrix

    bone_names = ['mixamorig:' + name.split(':')[-1] for name in bone_names]
    np.savez(
        args.output,
        bone_names=bone_names,
        bone_rest_mats=bone_rest_mats,
        bone_pose_mats=bone_pose_mats,
        frametime=frametime
    )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_fbx', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    main(args)
