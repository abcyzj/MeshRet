import sys
from argparse import ArgumentParser, Namespace

import bmesh
import bpy
import numpy as np


def extract_mesh_data(me):
    """
    Extract skinning weight from a given mesh
    """
    verts = me.data.vertices
    edges = me.data.edges
    faces = me.data.polygons
    vgrps = me.vertex_groups

    vert_cors = np.zeros([len(verts), 3], dtype=np.float32)
    edge_links = np.zeros([len(edges), 2], dtype=int)
    face_polygons = np.zeros([len(faces), 3], dtype=int)
    weight = np.zeros((len(verts), len(vgrps)), dtype=np.float32)
    vgrp_label = vgrps.keys()

    for i, vert in enumerate(verts):
        vert_cors[i] = vert.co
        for g in vert.groups:
            j = g.group
            weight[i, j] = g.weight

    for i, edge in enumerate(edges):
        edge_links[i] = edge.vertices

    for i, face in enumerate(faces):
        face_polygons[i] = face.vertices


    return vert_cors, edge_links, face_polygons, weight, vgrp_label


def merge_mesh_data(meshes):
    """
    Merge multiple meshes into one
    """
    vert_cors = []
    edge_links = []
    face_polygons = []
    weight = []
    vgrp_label = []
    for me in meshes:
        v, e, f, w, l = extract_mesh_data(me)
        vert_cors.append(v)
        edge_links.append(e)
        face_polygons.append(f)
        weight.append(w)
        vgrp_label.append(l)

    vgrp_label_set = set()
    merged_vgrp_label = []
    for l in vgrp_label:
        for name in l:
            if name not in vgrp_label_set:
                merged_vgrp_label.append(name)
        vgrp_label_set.update(l)

    v_count = 0
    for v, e, f in zip(vert_cors, edge_links, face_polygons):
        e += v_count
        f += v_count
        v_count += v.shape[0]
    merged_vert_cors = np.concatenate(vert_cors, axis=0)
    merged_edge_links = np.concatenate(edge_links, axis=0)
    merged_face_polygons = np.concatenate(face_polygons, axis=0)

    merged_weight = np.zeros((merged_vert_cors.shape[0], len(merged_vgrp_label)), dtype=weight[0].dtype)
    v_count = 0
    for v, w, l in zip(vert_cors, weight, vgrp_label):
        for i, name in enumerate(l):
            j = merged_vgrp_label.index(name)
            merged_weight[v_count:v_count+v.shape[0], j] = w[:, i]
        v_count += v.shape[0]

    return merged_vert_cors, merged_edge_links, merged_face_polygons, merged_weight, merged_vgrp_label


def extract_arm_data(arm, vgrp_label, bone_adjust_matrix=None):
    bones = arm.data.bones
    bone_names = [b.name for b in bones]
    bone_parents = np.zeros([len(bones)], dtype=int)
    bone_cors = np.zeros([len(bones), 3], dtype=np.float32)
    bone_rest_mat = np.zeros([len(bones), 3, 3], dtype=np.float32)
    for bone_idx, bone in enumerate(bones):
        if bone.parent is None:
            bone_parents[bone_idx] = -1
        else:
            bone_parents[bone_idx] = bone_names.index(bone.parent.name)
        if bone_adjust_matrix is not None:
            bone_cors[bone_idx] = bone_adjust_matrix @ bone.matrix_local.translation
            bone_rest_mat[bone_idx] = bone_adjust_matrix @ bone.matrix_local.to_3x3()
        else:
            bone_cors[bone_idx] = bone.matrix_local.translation
            bone_rest_mat[bone_idx] = bone.matrix_local.to_3x3()
    for name in vgrp_label:
        bone_idx = bone_names.index(name)
    return bone_names, bone_cors, bone_parents, bone_rest_mat


def rearrange_vgrp(vgrp_label, weight, bone_names, bone_cors, bone_parents, bone_rest_mats):
    '''
    rearragne vgrp order according to bone order
    '''
    assert len(vgrp_label) <= len(bone_names)

    new_vgrp_label = bone_names.copy()
    new_vgrp_cors = bone_cors.copy()
    new_vgrp_parents = bone_parents.copy()
    new_vgrp_rest_mats = bone_rest_mats.copy()
    new_weight = np.zeros([weight.shape[0], len(bone_names)], dtype=weight.dtype)

    for i, name in enumerate(bone_names):
        try:
            j = vgrp_label.index(name)
        except ValueError:
            new_weight[:, i] = 0.0
            continue
        new_weight[:, i] = weight[:, j]

    return new_vgrp_label, new_vgrp_cors, new_vgrp_parents, new_vgrp_rest_mats, new_weight


def extract_t_pose(arm, vgrp_label, vgrp_parents):
    '''
    extract possible t-pose from posed armature
    '''
    pose_bones = [arm.pose.bones[name] for name in vgrp_label]
    t_pose = np.zeros([len(vgrp_label), 3, 3], dtype=np.float32)
    for j, pose_bone in enumerate(pose_bones):
        rest_bone = pose_bone.bone
        if vgrp_parents[j] == -1:
            t_pose[j] = pose_bone.matrix.to_3x3() @ rest_bone.matrix_local.to_3x3().inverted()
        else:
            parent_bone = pose_bones[vgrp_parents[j]]
            parent_rest_bone = parent_bone.bone
            t_pose[j] = parent_rest_bone.matrix_local.to_3x3() @ parent_bone.matrix.to_3x3().inverted() @ pose_bone.matrix.to_3x3() @ rest_bone.matrix_local.to_3x3().inverted()
    return t_pose


def main(args: Namespace):
    bpy.ops.import_scene.fbx(filepath=args.input, automatic_bone_orientation=True)

    meshes = []
    arm = None
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            bm = bmesh.new()
            bm.from_mesh(obj.data)
            bmesh.ops.triangulate(bm, faces=bm.faces)
            bm.to_mesh(obj.data)
            bm.free()
            meshes.append(obj)
        if obj.type == 'ARMATURE':
            arm = obj


    vert_cors, edge_links, face_polygons, weight, vgrp_label = merge_mesh_data(meshes)
    if args.adjust_bone:
        bone_adjust_matrix = meshes[0].matrix_world.to_3x3().normalized().inverted()
    else:
        bone_adjust_matrix = None
    bone_names, bone_cors, bone_parents, bone_rest_mats = extract_arm_data(arm, vgrp_label, bone_adjust_matrix)
    vgrp_label, vgrp_cors, vgrp_parents, vgrp_rest_mats, weight = rearrange_vgrp(vgrp_label, weight, bone_names, bone_cors, bone_parents, bone_rest_mats)
    t_pose = extract_t_pose(arm, vgrp_label, vgrp_parents)
    vgrp_label = ['mixamorig:'+name.split(':')[-1] for name in vgrp_label]
    np.savez(
        args.output,
        verts=vert_cors,
        edges=edge_links,
        faces=face_polygons,
        weight=weight,
        vgrp_label=vgrp_label,
        vgrp_cors=vgrp_cors,
        vgrp_parents=vgrp_parents,
        vgrp_rest_mats=vgrp_rest_mats,
        t_pose=t_pose
        )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--adjust_bone', action='store_true')

    argv = sys.argv
    args = parser.parse_args(argv[argv.index('--')+1:])

    main(args)
