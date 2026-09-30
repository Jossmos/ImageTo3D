"""
Blender script voor het verwerken van een bestaande .blend file.
Snijdt de vorm uit met Geometry Nodes op basis van een alpha map en maakt het massief.

Gebruik:
  blender <file.blend> -b -P blender_apply_alpha_solidify.py -- <alpha_map.png> <output.stl>
"""

import sys
import os
import bpy
import bmesh
from pathlib import Path

def log_info(msg):
    print(f"[INFO] {msg}")
    sys.stdout.flush()

def log_error(msg):
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.stderr.flush()

def main():
    if "--" in sys.argv:
        args = sys.argv[sys.argv.index("--") + 1:]
    else:
        args = sys.argv[1:]

    if len(args) < 2:
        log_error("Onvoldoende argumenten. Verwacht: <alpha_map> <output_stl>")
        sys.exit(1)

    alpha_path = str(Path(args[0]).resolve())
    output_stl = str(Path(args[1]).resolve())
    
    # Opties (kan later worden uitgebreid met argparse/env vars)
    bottom_z = -2.0
    alpha_threshold = 0.1

    log_info("="*60)
    log_info("Start 2e Pipeline: Blend -> Solid STL")
    log_info(f"Alpha Map: {alpha_path}")
    log_info(f"Output STL: {output_stl}")
    log_info("="*60)

    if not os.path.isfile(alpha_path):
        log_error(f"Alpha map niet gevonden: {alpha_path}")
        sys.exit(1)

    # 1. Zoek het te bewerken object
    # Neem de active object, of zoek een mesh genaamd 'DisplacementPlane', of pak de eerste beste mesh
    obj = bpy.context.active_object
    if not obj or obj.type != 'MESH':
        for o in bpy.data.objects:
            if o.type == 'MESH':
                obj = o
                break

    if not obj:
        log_error("Geen mesh object gevonden in het .blend bestand!")
        sys.exit(1)

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    log_info(f"Werken met object: {obj.name}")

    # 2. Pas eventuele bestaande modifiers toe (zodat we een dichte mesh hebben voor GN)
    for mod in list(obj.modifiers):
        try:
            bpy.ops.object.modifier_apply(modifier=mod.name)
            log_info(f"Bestaande modifier toegepast: {mod.name}")
        except Exception as e:
            log_info(f"Kon modifier '{mod.name}' niet toepassen: {e}")

    # 3. Laad de alpha image
    try:
        alpha_image = bpy.data.images.load(alpha_path, check_existing=False)
        log_info("Alpha map ingeladen.")
    except Exception as e:
        log_error(f"Fout bij laden alpha map: {e}")
        sys.exit(1)

    # 4. Voeg Geometry Nodes Cutout toe
    log_info("Geometry Nodes instellen voor Alpha Cutout...")
    node_tree = bpy.data.node_groups.new(name="AlphaCutout", type="GeometryNodeTree")
    nodes = node_tree.nodes
    links = node_tree.links

    # Voeg sockets toe aan de Node Tree interface (nodig sinds Blender 4.0+)
    try:
        node_tree.interface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
        node_tree.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    except AttributeError:
        try:
            node_tree.inputs.new('NodeSocketGeometry', "Geometry")
            node_tree.outputs.new('NodeSocketGeometry', "Geometry")
        except Exception:
            pass

    geo_in = nodes.new(type="NodeGroupInput")
    geo_out = nodes.new(type="NodeGroupOutput")

    img_node = nodes.new(type="GeometryNodeImageTexture")
    try:
        img_node.extension = 'CLIP'
    except Exception:
        pass
    try:
        img_node.image = alpha_image
    except Exception:
        try:
            img_node.inputs["Image"].default_value = alpha_image
        except Exception as e:
            log_error(f"Kan alpha image niet koppelen aan Geometry Nodes: {e}")

    # Haal UV map op zodat de afbeelding exact 1x over het model wordt geprojecteerd
    uv_node = nodes.new(type="GeometryNodeInputNamedAttribute")
    uv_node.data_type = 'FLOAT_VECTOR'
    uv_name = "UVMap"
    if obj.data.uv_layers.active:
        uv_name = obj.data.uv_layers.active.name
    try:
        uv_node.inputs["Name"].default_value = uv_name
    except Exception:
        pass

    math_node = nodes.new(type="FunctionNodeCompare")
    try:
        math_node.operation = 'LESS_THAN'
        math_node.data_type = 'FLOAT'
        math_node.inputs[1].default_value = alpha_threshold
    except Exception:
        # Fallback voor oudere versies
        math_node = nodes.new(type="ShaderNodeMath")
        math_node.operation = "LESS_THAN"
        math_node.inputs[1].default_value = alpha_threshold

    delete_node = nodes.new(type="GeometryNodeDeleteGeometry")
    delete_node.domain = 'FACE'

    # Verbinden
    try:
        try:
            links.new(geo_in.outputs["Geometry"], delete_node.inputs["Geometry"])
        except Exception:
            links.new(geo_in.outputs[0], delete_node.inputs[0])
            
        # Koppel UV map aan Image Texture (Vector input)
        try:
            links.new(uv_node.outputs["Attribute"], img_node.inputs["Vector"])
        except Exception:
            try:
                links.new(uv_node.outputs[0], img_node.inputs[0])
            except Exception:
                pass
        
        # Probeer Alpha output te verbinden, anders Color/Value
        try:
            links.new(img_node.outputs["Alpha"], math_node.inputs[0])
        except Exception:
            links.new(img_node.outputs[0], math_node.inputs[0])
            
        try:
            links.new(math_node.outputs.get("Value", math_node.outputs[0]), delete_node.inputs["Selection"])
        except Exception:
            links.new(math_node.outputs[0], delete_node.inputs[1])

        try:
            links.new(delete_node.outputs["Geometry"], geo_out.inputs["Geometry"])
        except Exception:
            links.new(delete_node.outputs[0], geo_out.inputs[0])
        
        # Voeg de modifier toe en pas toe
        gn_mod = obj.modifiers.new(name="AlphaCutoutModifier", type='NODES')
        gn_mod.node_group = node_tree
        
        # Apply modifier
        bpy.ops.object.modifier_apply(modifier=gn_mod.name)
        log_info("Geometry Nodes Alpha Cutout succesvol toegepast!")
        
    except Exception as e:
        log_error(f"Fout tijdens Geometry Nodes setup/apply: {e}")
        sys.exit(1)

    # 5. Massief maken via BMesh (de robuuste fix)
    log_info("Massief maken via BMesh extrusie...")
    bpy.ops.object.mode_set(mode='OBJECT')
    
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.normal_update()

    boundary_edges = [e for e in bm.edges if not e.is_manifold]
    if boundary_edges:
        try:
            # Extrude randen
            extrude_res = bmesh.ops.extrude_edge_only(bm, edges=boundary_edges)
            
            # Verplaats naar bodem (Z = bottom_z)
            new_verts = [v for v in extrude_res['geom'] if isinstance(v, bmesh.types.BMVert)]
            for v in new_verts:
                v.co.z = float(bottom_z)
                
            # Vul het gat aan de onderkant
            bottom_edges = [e for e in bm.edges if not e.is_manifold]
            if bottom_edges:
                log_info("  > Gaten vullen via triangle_fill...")
                try:
                    bmesh.ops.triangle_fill(bm, use_beauty=False, use_dissolve=False, edges=bottom_edges)
                except Exception as fill_e:
                    log_info(f"  > triangle_fill mislukt: {fill_e}, probeer holes_fill...")
                    try:
                        bmesh.ops.holes_fill(bm, edges=bottom_edges, sides=100000)
                    except Exception as edge_e:
                        log_info(f"  > holes_fill mislukt: {edge_e}")

            # Triangulate
            bmesh.ops.triangulate(bm, faces=list(bm.faces))
            # Fix normals
            bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
            
            log_info("Mesh is succesvol massief gemaakt en gesloten.")
            
        except Exception as e:
            log_error(f"Fout tijdens massief maken: {e}")
    else:
        log_info("Geen open randen gevonden. Object was mogelijk al massief.")

    # Schrijf terug naar mesh
    bm.normal_update()
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()

    # 6. Exporteren naar STL
    log_info("Exporteren naar STL...")
    try:
        # Smooth shading
        bpy.ops.object.shade_smooth()
        
        # Normals naar buiten dwingen net voor export
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode='OBJECT')
        
        os.makedirs(os.path.dirname(output_stl), exist_ok=True)
        
        try:
            bpy.ops.wm.stl_export(filepath=output_stl, export_selected_objects=True)
            log_info(f"Export succesvol (wm.stl_export): {output_stl}")
        except Exception:
            bpy.ops.export_mesh.stl(
                filepath=output_stl,
                use_selection=True,
                use_mesh_modifiers=True
            )
            log_info(f"Export succesvol (export_mesh.stl): {output_stl}")
    except Exception as e:
        log_error(f"Fout bij export naar STL: {e}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_error(f"Fatale fout: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
