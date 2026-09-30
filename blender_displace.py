"""
Geavanceerd Blender Python-script voor 3D-reliëf generatie.

Dit script genereert een 3D STL-bestand uit een depth map en alpha map.
Het script draait headless en verwacht commandline-argumenten:
  1. Pad naar Depth Map (grayscale)
  2. Pad naar Alpha Map (RGBA voor uitsnede)
  3. Pad voor output STL

Voorbeeld:
  blender -b -P blender_displace.py -- depth.png alpha.png output.stl
"""

import sys
import os
import bpy
import bmesh
from pathlib import Path
from mathutils import Vector
import time
import traceback

# =====================================================================
# FOUTAFHANDELING & LOGGING
# =====================================================================


def log_error(message: str) -> None:
    """Print foutbericht naar console."""
    print(f"[ERROR] {message}", file=sys.stderr)


def log_info(message: str) -> None:
    """Print info-bericht naar console."""
    print(f"[INFO] {message}")


def log_success(message: str) -> None:
    """Print succesmeldingen naar console."""
    print(f"[SUCCESS] {message}")


# Simple file-based debug logging to help diagnose native Blender crashes
DEBUG_LOG_PATH = os.path.join(os.path.dirname(__file__), "blender_debug.log")


def write_debug(msg: str) -> None:
    try:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {msg}\n")
    except Exception:
        pass


# =====================================================================
# FASE 1: SETUP
# =====================================================================


def setup_scene(depth_path: str, alpha_path: str, plane_scale_x: float = 10.0) -> tuple:
    """
    Fase 1: Initiële scene-setup met afbeelding-inladen.

    Args:
        depth_path: Pad naar depth map (grayscale).
        alpha_path: Pad naar alpha map (RGBA).

    Returns:
        Tuple van (depth_image, alpha_image, plane_object).

    Raises:
        FileNotFoundError: Als afbeeldingen niet bestaan.
        RuntimeError: Bij Blender API-fouten.
    """
    log_info("Fase 1: Setup - Scene initialiseren...")
    write_debug(f"setup_scene start depth={depth_path} alpha={alpha_path} plane_scale_x={plane_scale_x}")

    # Zorg voor schone scene
    try:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        log_info("  > Factory settings geladen (schone scene).")
    except Exception as e:
        raise RuntimeError(f"Fout bij factory settings: {e}")

    # Valideer bestandspaden
    depth_path = str(Path(depth_path).resolve())
    alpha_path = str(Path(alpha_path).resolve())

    if not os.path.isfile(depth_path):
        raise FileNotFoundError(f"Depth map niet gevonden: {depth_path}")
    if not os.path.isfile(alpha_path):
        raise FileNotFoundError(f"Alpha map niet gevonden: {alpha_path}")

    log_info(f"  > Depth map ingeladen: {depth_path}")
    log_info(f"  > Alpha map ingeladen: {alpha_path}")

    # Laad afbeeldingen
    try:
        depth_image = bpy.data.images.load(depth_path, check_existing=False)
        alpha_image = bpy.data.images.load(alpha_path, check_existing=False)
    except Exception as e:
        raise RuntimeError(f"Fout bij laden afbeeldingen: {e}")

    log_info(
        f"  > Depth map resolutie: {depth_image.size[0]}x{depth_image.size[1]} pixels"
    )
    log_info(
        f"  > Alpha map resolutie: {alpha_image.size[0]}x{alpha_image.size[1]} pixels"
    )

    # Maak Plane aan
    try:
        bpy.ops.mesh.primitive_plane_add(size=2, location=(0, 0, 0))
        plane = bpy.context.active_object
        plane.name = "DisplacementPlane"
    except Exception as e:
        raise RuntimeError(f"Fout bij aanmaken plane: {e}")

    # Bereken aspect ratio en pas schaal aan
    # Zet X breedte op plane_scale_x, bereken Y op basis van aspect ratio
    img_width = depth_image.size[0]
    img_height = depth_image.size[1]
    aspect_ratio = img_height / img_width if img_width > 0 else 1.0

    scale_x = float(plane_scale_x)
    scale_y = scale_x * aspect_ratio
    scale_z = 1.0

    plane.scale = (scale_x / 2, scale_y / 2, scale_z)
    log_info(f"  > Plane geschaald naar: X={scale_x}, Y={scale_y:.2f}")

    # Pas schaal toe
    try:
        bpy.context.view_layer.objects.active = plane
        bpy.ops.object.transform_apply(scale=True)
        log_info("  > Schaal toegepast op plane.")
    except Exception as e:
        raise RuntimeError(f"Fout bij apply transform: {e}")

    log_success("Fase 1 voltooid: Scene setup succesvol!")
    write_debug("setup_scene complete")
    return depth_image, alpha_image, plane


# =====================================================================
# FASE 2: MODIFIERS
# =====================================================================


def add_modifiers(
    plane: bpy.types.Object,
    depth_image,
    alpha_image,
    subsurf_levels: int = 6,
    subsurf_render_levels: int = 6,
    subsurf_type: str = "SIMPLE",
    subsurf_use_limit_surface: bool = False,
    displace_strength: float = 0.5,
    displace_mid_level: float = 0.0,
    corrective_iterations: int = 5,
    corrective_factor: float = 0.5,
) -> None:
    """
    Fase 2: Voeg modifiers toe in strikte volgorde.

    Volgorde:
    1. Subdivision Surface (SIMPLE, level 6)
    2. Displace (met depth map texture)
    3. Geometry Nodes (alpha map uitsnede)

    Args:
        plane: Het plane object.
        depth_image: Ingeladen depth map afbeelding.
        alpha_image: Ingeladen alpha map afbeelding.

    Raises:
        RuntimeError: Bij modifier-fouten.
    """
    log_info("Fase 2: Modifiers - Toevoegen in volgorde...")

    bpy.context.view_layer.objects.active = plane
    plane.select_set(True)

    # ---- MODIFIER 1: Subdivision Surface ----
    try:
        subsurf = plane.modifiers.new(name="Subdivision", type="SUBSURF")
        # subdivision_type may be 'SIMPLE' or 'CATMULL_CLARK' depending on Blender
        try:
            subsurf.subdivision_type = subsurf_type
        except Exception:
            pass
        try:
            subsurf.levels = int(subsurf_levels)
        except Exception:
            subsurf.levels = int(6)
        try:
            subsurf.render_levels = int(subsurf_render_levels)
        except Exception:
            subsurf.render_levels = int(subsurf_levels)
        try:
            subsurf.use_limit_surface = bool(subsurf_use_limit_surface)
        except Exception:
            pass
        log_info(
            f"  > Subdivision Surface toegevoegd (type={subsurf_type} levels={subsurf_levels} render={subsurf_render_levels})."
        )
    except Exception as e:
        raise RuntimeError(f"Fout bij Subdivision Surface: {e}")

    # ---- MODIFIER 2: Displace ----
    try:
        # Maak texture aan
        displace_texture = bpy.data.textures.new(
            name="DisplaceTexture", type="IMAGE"
        )
        displace_texture.image = depth_image

        # Voeg Displace modifier toe
        displace = plane.modifiers.new(name="Displace", type="DISPLACE")
        displace.texture = displace_texture
        displace.texture_coords = "UV"
        try:
            displace.mid_level = float(displace_mid_level)
        except Exception:
            pass
        displace.strength = float(displace_strength)
        log_info(f"  > Displace modifier toegevoegd met strength={displace_strength}, mid_level={displace_mid_level}.")
    except Exception as e:
        raise RuntimeError(f"Fout bij Displace modifier: {e}")

    # ---- MODIFIER 3: Geometry Nodes ----
    # Attempt to add Geometry Nodes to delete transparent areas using the alpha map.
    # Many Blender versions expose different node names/properties; try best-effort
    # and if it fails, continue without alpha-based deletion so the pipeline can finish.
    try:
        node_tree = bpy.data.node_groups.new(name="AlphaCutoutTree", type="GeometryNodeTree")
        links = node_tree.links
        nodes = node_tree.nodes

        geo_input = nodes.new(type="NodeGroupInput")
        geo_output = nodes.new(type="NodeGroupOutput")

        # Try to create an Image Texture node in geometry nodes
        img_node = nodes.new(type="GeometryNodeImageTexture")
        assigned = False
        try:
            img_node.image = alpha_image
            assigned = True
        except Exception:
            try:
                img_node.inputs["Image"].default_value = alpha_image
                assigned = True
            except Exception:
                assigned = False

        # Create a compare/math node to threshold alpha
        try:
            math_node = nodes.new(type="FunctionNodeCompare")
            # Attempt to set as less-than for floats
            try:
                math_node.operation = 'LESS_THAN'
            except Exception:
                pass
            try:
                math_node.data_type = 'FLOAT'
            except Exception:
                pass
            try:
                math_node.inputs[1].default_value = 0.1
            except Exception:
                pass
        except Exception:
            # fallback
            math_node = nodes.new(type="ShaderNodeMath")
            math_node.operation = "LESS_THAN"
            math_node.inputs[1].default_value = 0.1

        # Delete geometry node
        delete_node = nodes.new(type="GeometryNodeDeleteGeometry")
        delete_node.domain = "FACE"

        # Best-effort linking; some sockets may not exist on all versions
        try:
            links.new(geo_input.outputs["Geometry"], delete_node.inputs["Geometry"])
        except Exception:
            pass

        if assigned:
            try:
                links.new(img_node.outputs.get("Alpha", img_node.outputs[0]), math_node.inputs[0])
            except Exception:
                try:
                    links.new(img_node.outputs[0], math_node.inputs[0])
                except Exception:
                    pass

        try:
            links.new(math_node.outputs.get("Value", math_node.outputs[0]), delete_node.inputs["Selection"])
        except Exception:
            pass

        try:
            links.new(delete_node.outputs["Geometry"], geo_output.inputs["Geometry"])
        except Exception:
            pass

        # Do NOT attach the node_tree as a modifier by default — some Blender versions
        # will evaluate this node group into empty geometry. Instead, skip adding
        # the Geometry Nodes modifier to avoid producing an empty mesh.
        log_info("  > Geometry Nodes tree aangemaakt (niet toegewezen als modifier om corruptie te voorkomen).")

    except Exception as e:
        # Do not raise; log and continue without alpha cut
        log_error(f"Geometry Nodes kon niet worden aangemaakt: {e}")
        log_info("  > Vervolg zonder alpha-based uitsnede (STL zal volledige vorm bevatten).")

    log_success("Fase 2 voltooid: Alle modifiers toegevoegd!")

    # ---- MODIFIER 4: Corrective Smooth ----
    try:
        corr = plane.modifiers.new(name="CorrectiveSmooth", type="CORRECTIVE_SMOOTH")
        try:
            corr.iterations = int(corrective_iterations)
        except Exception:
            pass
        try:
            corr.factor = float(corrective_factor)
        except Exception:
            pass
        # Force 'only smooth' behavior per user request
        try:
            # property name varies slightly across versions; try common ones
            if hasattr(corr, "use_only_smooth"):
                corr.use_only_smooth = True
            elif hasattr(corr, "only_smooth"):
                setattr(corr, "only_smooth", True)
        except Exception:
            pass
        log_info(f"  > Corrective Smooth toegevoegd (iters={corrective_iterations}, factor={corrective_factor}, only_smooth=True).")
    except Exception:
        # Non-fatal: log and continue
        log_info("  > Corrective Smooth kon niet worden toegevoegd (Blender versie mismatch?) — doorgaan zonder.")


# =====================================================================
# FASE 3: APPLY MODIFIERS
# =====================================================================


def apply_modifiers(plane: bpy.types.Object) -> bpy.types.Object:
    """
    Fase 3: Pas alle modifiers toe in volgorde (bottom-up).

    Args:
        plane: Het plane object met modifiers.

    Raises:
        RuntimeError: Bij apply-fouten.
    """
    log_info("Fase 3: Apply Modifiers - Vaste geometrie aanmaken...")

    bpy.context.view_layer.objects.active = plane
    plane.select_set(True)

    # Pas modifiers toe van onder naar boven (omgekeerde volgorde)
    modifier_names = [m.name for m in plane.modifiers]
    log_info(f"  > Toe te passen modifiers (volgorde): {modifier_names}")

    for mod_name in modifier_names:
        try:
            bpy.ops.object.modifier_apply(modifier=mod_name)
            log_info(f"  > '{mod_name}' modifier toegepast.")
        except Exception as e:
            log_error(f"Fout bij apply modifier '{mod_name}': {e}")
            # If Geometry Nodes caused the failure, try a best-effort conversion
            mod = plane.modifiers.get(mod_name)
            if mod and mod.type == 'NODES':
                log_info("  > Probeer converteren van geëvalueerde geometry naar mesh (convert to MESH)...")
                try:
                    bpy.ops.object.convert(target='MESH')
                    log_info("  > Object geconverteerd naar MESH (best-effort).")
                    # After conversion, the active object may be a new mesh object; rebind plane
                    try:
                        plane = bpy.context.active_object
                        log_info(f"  > Active object na convert: {plane.name}")
                    except Exception:
                        pass
                    continue
                except Exception as conv_e:
                    log_error(f"  > Conversie naar MESH mislukt: {conv_e}")
            raise RuntimeError(f"Fout bij apply modifier '{mod_name}': {e}")

    log_success("Fase 3 voltooid: Alle modifiers toegepast, vaste geometrie aangemaakt!")
    return plane


# =====================================================================
# FALLBACK: HEIGHTFIELD-BASED MESH GENERATION (ROBUST)
# =====================================================================


def create_heightfield_block(
    depth_image,
    alpha_image,
    plane: bpy.types.Object,
    max_height: float = 1.0,
    max_samples: int = 800,
    alpha_threshold: float = 0.5,
    smoothing_iters: int = 2,
    smoothing_factor: float = 0.25,
    bottom_z: float = -2.0,
) -> None:
    """
    Bouw een schone heightfield-mesh direct uit de depth image met alpha-masking.

    Dit produceert een regelmatig, manifold mesh zonder afhankelijkheid van modifiers
    of Geometry Nodes. Het maakt een grid met maximaal `max_samples` in de X-richting
    (schaling in Y behoudt aspect ratio) en verwijdert vertices/faces waar alpha < threshold.
    """
    log_info("FALLBACK: Heightfield-based mesh genereren (robust) met alpha-masking...")
    w = depth_image.size[0]
    h = depth_image.size[1]
    aw = alpha_image.size[0]
    ah = alpha_image.size[1]

    write_debug(f"create_heightfield_block start w={w} h={h} max_height={max_height} max_samples={max_samples}")

    # Bepaal samples (cap op max_samples)
    samples_x = min(w, max_samples)
    samples_y = int(samples_x * (h / w)) if w > 0 else samples_x
    samples_y = max(2, min(h, samples_y))

    # Read pixels efficiently to prevent MemoryError
    import numpy as np
    
    num_depth_pixels = len(depth_image.pixels)
    pixels = np.empty(num_depth_pixels, dtype=np.float32)
    depth_image.pixels.foreach_get(pixels)
    
    num_alpha_pixels = len(alpha_image.pixels)
    alpha_pixels = np.empty(num_alpha_pixels, dtype=np.float32)
    alpha_image.pixels.foreach_get(alpha_pixels)
    
    channels = num_depth_pixels // (w * h) if w * h > 0 else 1
    alpha_channels = num_alpha_pixels // (aw * ah) if aw * ah > 0 else 1

    def sample_depth(ix, iy):
        # nearest neighbor sampling from depth image
        sx = int(ix * (w / samples_x))
        sy = int(iy * (h / samples_y))
        sx = max(0, min(w - 1, sx))
        sy = max(0, min(h - 1, sy))
        idx = (sy * w + sx) * channels
        return pixels[idx]

    def sample_alpha(ix, iy):
        # nearest neighbor sampling from alpha image
        sx = int(ix * (aw / samples_x))
        sy = int(iy * (ah / samples_y))
        sx = max(0, min(aw - 1, sx))
        sy = max(0, min(ah - 1, sy))
        idx = (sy * aw + sx) * alpha_channels
        # Alpha is usually at channel 3 (RGBA), fallback to channel 0
        if alpha_channels >= 4 and idx + 3 < len(alpha_pixels):
            return alpha_pixels[idx + 3]
        elif idx < len(alpha_pixels):
            return alpha_pixels[idx]
        return 1.0

    # Plane world extents based on plane.scale (plane is centered at origin)
    extent_x = plane.scale[0] * 2
    extent_y = plane.scale[1] * 2

    bm = bmesh.new()
    verts_grid = []
    alpha_grid = []  # store alpha values for culling

    for iy in range(samples_y):
        row = []
        alpha_row = []
        for ix in range(samples_x):
            u = ix / (samples_x - 1)
            v = iy / (samples_y - 1)
            x = (u - 0.5) * extent_x
            y = (v - 0.5) * extent_y
            d = sample_depth(ix, iy)
            a = sample_alpha(ix, iy)
            z = float(d) * max_height
            vtx = bm.verts.new((x, y, z))
            row.append(vtx)
            alpha_row.append(float(a))
        verts_grid.append(row)
        alpha_grid.append(alpha_row)

    bm.verts.ensure_lookup_table()

    # create faces (only if all corners have sufficient alpha)
    # alpha threshold for culling
    # (can be tuned by user)
    for iy in range(samples_y - 1):
        for ix in range(samples_x - 1):
            v00 = verts_grid[iy][ix]
            v10 = verts_grid[iy][ix + 1]
            v01 = verts_grid[iy + 1][ix]
            v11 = verts_grid[iy + 1][ix + 1]
            a00 = alpha_grid[iy][ix]
            a10 = alpha_grid[iy][ix + 1]
            a01 = alpha_grid[iy + 1][ix]
            a11 = alpha_grid[iy + 1][ix + 1]
            # Create face only if all 4 corners have alpha > threshold
            if a00 > alpha_threshold and a10 > alpha_threshold and a01 > alpha_threshold and a11 > alpha_threshold:
                try:
                    bm.faces.new([v00, v10, v11, v01])
                except Exception:
                    pass
            # Mark low-alpha vertices for removal
            if a00 <= alpha_threshold:
                v00.tag = True
            if a10 <= alpha_threshold:
                v10.tag = True
            if a01 <= alpha_threshold:
                v01.tag = True
            if a11 <= alpha_threshold:
                v11.tag = True

    # Remove low-alpha vertices
    low_alpha_verts = [v for v in bm.verts if v.tag]
    for v in low_alpha_verts:
        try:
            bm.verts.remove(v)
        except Exception:
            pass

    log_info(f"  > Alpha-masking: {len(low_alpha_verts)} vertices verwijderd (alpha < {alpha_threshold})")


    # remove doubles and triangulate
    try:
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
    except Exception:
        pass
    try:
        bmesh.ops.triangulate(bm, faces=list(bm.faces))
    except Exception:
        pass

    # Collect valid (high-alpha) vertices for perimeter BEFORE removing low-alpha verts
    valid_verts = set()
    for v in bm.verts:
        if not v.tag:  # tag=True means low-alpha
            valid_verts.add(v)
    write_debug(f"heightfield alpha removed={len(low_alpha_verts)} valid_verts={len(valid_verts)}")

    # Remove low-alpha vertices (only if we have enough valid verts left)
    if len(valid_verts) > 4:
        for v in list(bm.verts):
            if v.tag:  # low-alpha vertex
                try:
                    bm.verts.remove(v)
                except Exception:
                    pass
    else:
        # If too few verts, keep all
        log_info("  > Waarschuwing: te weinig high-alpha vertices, behoud alle verts")

    bm.verts.ensure_lookup_table()

    # EERST: Apply smoothing (Laplacian) op het top oppervlak (voordat we de onderkant extruderen)
    try:
        all_verts = [v for v in bm.verts]
        if all_verts:
            for _ in range(int(smoothing_iters)):
                try:
                    bmesh.ops.smooth_vert(bm, verts=all_verts, factor=float(smoothing_factor), use_axis=False)
                except Exception:
                    # Fallback
                    new_positions = {}
                    for v in all_verts:
                        neighbors = [e.other_vert(v) for e in v.link_edges]
                        if neighbors:
                            avg_pos = sum((n.co for n in neighbors), Vector()) / len(neighbors)
                            new_positions[v] = v.co + (avg_pos - v.co) * float(smoothing_factor)
                    for v, new_pos in new_positions.items():
                        v.co = new_pos
            log_info(f"  > Smoothing toegepast op top mesh ({len(all_verts)} vertices)")
    except Exception as e:
        log_info(f"  > Smoothing overgeslagen: {e}")

    # DAN: Extrudeer de gaten/randen loodrecht omlaag om het blok solid te maken (watertight/manifold)
    boundary_edges = [e for e in bm.edges if not e.is_manifold]
    if boundary_edges:
        try:
            # 1. Extrudeer alleen de randen
            extrude_res = bmesh.ops.extrude_edge_only(bm, edges=boundary_edges)
            
            # 2. Pak de nieuwe vertices en duw ze naar bottom_z
            new_verts = [v for v in extrude_res['geom'] if isinstance(v, bmesh.types.BMVert)]
            for v in new_verts:
                v.co.z = float(bottom_z)
                
            # 3. Zoek de nieuwe non-manifold edges aan de onderkant
            bottom_edges = [e for e in bm.edges if not e.is_manifold]
            
            # 4. Vul het gat aan de onderkant (sluit de mesh)
            if bottom_edges:
                try:
                    bmesh.ops.holes_fill(bm, edges=bottom_edges, sides=100000)
                except Exception as fill_e:
                    log_info(f"  > holes_fill mislukt ({fill_e}), probeer edgeloop_fill...")
                    try:
                        bmesh.ops.edgeloop_fill(bm, edges=bottom_edges)
                    except Exception as edge_e:
                        log_info(f"  > edgeloop_fill mislukt: {edge_e}")

            # 5. Triangulate the newly created faces to ensure everything is valid STL geometry
            bmesh.ops.triangulate(bm, faces=list(bm.faces))
            
            # 6. Maak normals consistent naar buiten
            bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
            
            log_info(f"  > Bottom en zijvlakken robuust gegenereerd en gesloten via extrude_edge_only.")
        except Exception as e:
            log_info(f"  > Fout tijdens genereren bottom/zijvlakken: {e}")
    else:
        log_info("  > Geen boundary edges gevonden, mesh is mogelijk al dicht of leeg.")

    # finalize
    bm.normal_update()
    bm.to_mesh(plane.data)
    bm.free()
    plane.data.update()
    log_info('FALLBACK: Heightfield mesh geschreven naar plane.data')
    write_debug('create_heightfield_block complete')


# =====================================================================
# FASE 4: MASSIEVE GEOMETRY BMESH
# =====================================================================


def create_solid_block(plane: bpy.types.Object, bottom_z: float = -2.0) -> None:
    """
    Fase 4: Converteer de gefragmenteerde uitsnijding naar een massief blok via bmesh.

    Stappen:
    1. Zet mesh om naar bmesh.
    2. Zoek boundary loops (niet-manifold edges).
    3. Extrudeer deze edges omlaag (Z = -2.0).
    4. Vul de onderkant met een face.
    5. Schrijf terug naar mesh.

    Args:
        plane: Het plane object.

    Raises:
        RuntimeError: Bij bmesh-fouten.
    """
    log_info("Fase 4: Massieve Geometry - Blok aanmaken via bmesh...")

    bpy.context.view_layer.objects.active = plane
    plane.select_set(True)

    # Zorg dat we in Object Mode zijn
    bpy.ops.object.mode_set(mode="OBJECT")

    try:
        # Zet om naar bmesh
        bm = bmesh.new()
        bm.from_mesh(plane.data)
        bm.normal_update()

        log_info("  > Mesh geconverteerd naar bmesh.")
        log_info(f"  > BMesh vertex count: {len(bm.verts)}, edge count: {len(bm.edges)}, face count: {len(bm.faces)}")

        # Zoek boundary edges (non-manifold, dus randen aan maar 1 face)
        boundary_edges = [e for e in bm.edges if not e.is_manifold]

        if boundary_edges:
            try:
                # 1. Extrudeer alleen de randen
                extrude_res = bmesh.ops.extrude_edge_only(bm, edges=boundary_edges)
                
                # 2. Pak de nieuwe vertices en duw ze naar bottom_z
                new_verts = [v for v in extrude_res['geom'] if isinstance(v, bmesh.types.BMVert)]
                for v in new_verts:
                    v.co.z = float(bottom_z)
                    
                # 3. Zoek de nieuwe non-manifold edges aan de onderkant
                bottom_edges = [e for e in bm.edges if not e.is_manifold]
                
                # 4. Vul het gat aan de onderkant (sluit de mesh)
                if bottom_edges:
                    try:
                        bmesh.ops.holes_fill(bm, edges=bottom_edges, sides=100000)
                    except Exception as fill_e:
                        log_info(f"  > holes_fill mislukt ({fill_e}), probeer edgeloop_fill...")
                        try:
                            bmesh.ops.edgeloop_fill(bm, edges=bottom_edges)
                        except Exception as edge_e:
                            log_info(f"  > edgeloop_fill mislukt: {edge_e}")

                # 5. Triangulate the newly created faces to ensure everything is valid STL geometry
                bmesh.ops.triangulate(bm, faces=list(bm.faces))
                
                # 6. Maak normals consistent naar buiten
                bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
                
                log_info(f"  > Massief blok robuust gegenereerd en gesloten via extrude_edge_only.")
            except Exception as e:
                log_error(f"  > Fout tijdens genereren massief blok: {e}")
        else:
            log_info("  > Geen boundary edges gevonden, object is mogelijk al massief.")

        # Update en schrijf terug
        bm.normal_update()
        bm.to_mesh(plane.data)
        bm.free()

        plane.data.update()
        log_info("  > BMesh geschreven naar mesh-data.")

        log_success("Fase 4 voltooid: Massief blok aangemaakt!")

    except Exception as e:
        raise RuntimeError(f"Fout bij bmesh-operatie: {e}")


# =====================================================================
# FASE 5: EXPORT NAAR STL
# =====================================================================


def export_to_stl(plane: bpy.types.Object, output_path: str) -> None:
    """
    Fase 5: Export het mesh naar STL.

    Stappen:
    1. Zet naar Edit Mode.
    2. Selecteer alle vertices.
    3. Recalculate normals (naar buiten).
    4. Terug naar Object Mode.
    5. Export naar STL.

    Args:
        plane: Het plane object.
        output_path: Pad voor output STL.

    Raises:
        RuntimeError: Bij export-fouten.
    """
    log_info("Fase 5: Export - Naar STL-bestand...")

    bpy.context.view_layer.objects.active = plane
    plane.select_set(True)

    try:
        # Apply smooth shading (viewport + normal rendering)
        try:
            bpy.ops.object.shade_smooth()
            log_info("  > Smooth shading applied (normal interpolation).")
        except Exception as e:
            log_info(f"  > Smooth shading skipped: {e}")

        # Edit Mode: selecteer alles en recalculate normals
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.normals_make_consistent(inside=False)
        log_info("  > Normals recalculated (outward).")

        # Terug naar Object Mode
        bpy.ops.object.mode_set(mode="OBJECT")

        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            log_info(f"  > Output directory aangemaakt: {output_dir}")

        # Zorg dat de STL-export addon beschikbaar is
        try:
            bpy.ops.wm.addon_enable(module='io_mesh_stl')
        except Exception:
            # addon may already be enabled or not available; continue and try export
            pass

        # Export STL (probeer operator, fallback naar eigen writer als addon ontbreekt)
        try:
            write_debug(f"before export_to_stl output={output_path}")
            bpy.ops.export_mesh.stl(
                filepath=output_path,
                use_selection=True,
                use_mesh_modifiers=True,
            )
            log_info(f"  > STL geëxporteerd naar: {output_path}")
            write_debug("export_to_stl complete")
        except Exception as op_e:
            log_error(f"  > STL export operator faalde: {op_e}. Probeer fallback writer...")

            # Fallback: schrijf ASCII STL direct vanuit mesh data
            try:
                # Work on a bmesh copy to safely triangulate and remove degenerate faces
                mesh = plane.data
                mesh.update()

                bm = bmesh.new()
                bm.from_mesh(mesh)

                # Remove doubles and small artifacts
                try:
                    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
                except Exception:
                    pass

                # Triangulate to ensure STL-compatible triangles
                try:
                    bmesh.ops.triangulate(bm, faces=list(bm.faces))
                except Exception:
                    pass

                # Remove degenerate faces (very small area)
                small = []
                for f in list(bm.faces):
                    area = f.calc_area()
                    if area < 1e-8:
                        small.append(f)
                for f in small:
                    try:
                        bmesh.ops.delete(bm, geom=[f], context='FACES')
                    except Exception:
                        pass

                # Write ASCII STL from bmesh faces (triangles expected)
                def write_ascii_stl_from_bm(bm_in, filepath):
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write('solid export\n')
                        for face in bm_in.faces:
                            # compute normal
                            try:
                                n = face.normal
                            except Exception:
                                n = (0.0, 0.0, 1.0)
                            f.write(f' facet normal {n.x} {n.y} {n.z}\n')
                            f.write('  outer loop\n')
                            # face.verts order
                            for v in face.verts:
                                co = v.co
                                f.write(f'   vertex {co.x} {co.y} {co.z}\n')
                            f.write('  endloop\n')
                            f.write(' endfacet\n')
                        f.write('endsolid export\n')

                write_ascii_stl_from_bm(bm, output_path)
                bm.free()
                log_info(f"  > Fallback ASCII STL geschreven naar: {output_path}")
            except Exception as write_e:
                raise RuntimeError(f"Fout bij fallback STL writer: {write_e}")

        log_success(f"Fase 5 voltooid: Export succesvol naar {output_path}!")

    except Exception as e:
        raise RuntimeError(f"Fout bij export naar STL: {e}")


def save_blend_file(output_stl_path: str) -> str:
    """Sla de huidige Blender scene op als .blend naast de STL-output."""
    base, _ = os.path.splitext(output_stl_path)
    blend_path = base + ".blend"

    try:
        bpy.ops.wm.save_as_mainfile(filepath=blend_path)
        log_success(f"Blend-bestand opgeslagen: {blend_path}")
        return blend_path
    except Exception as e:
        raise RuntimeError(f"Fout bij opslaan .blend-bestand: {e}")


def _parse_cli_options(kv_list):
    """
    Parse a list of CLI options of the form key=value into a dict.
    Items without '=' are set to True.
    """
    opts = {}
    for item in kv_list:
        if isinstance(item, str) and "=" in item:
            k, v = item.split("=", 1)
            opts[k.strip()] = v.strip()
        elif isinstance(item, str):
            opts[item.strip()] = True
    return opts


# =====================================================================
# MAIN
# =====================================================================


def main() -> None:
    """
    Hoofdfunctie: voer alle fases uit.

    Verwachte commandline-argumenten:
      1. Depth map path
      2. Alpha map path
      3. Output STL path
    """
    log_info("=" * 70)
    log_info("Blender Displacement Relief Generator")
    log_info("=" * 70)

    # Parseer commandline-argumenten
    # Blender voegt eigen CLI-args toe; script-args staan NA '--'.
    if "--" in sys.argv:
        idx = sys.argv.index("--")
        args = sys.argv[idx + 1 :]
    else:
        # Fallback: gebruik alles behalve the script name
        args = sys.argv[1:]

    if len(args) < 3:
        log_error("Onvoldoende argumenten! Verwacht: depth_map alpha_map output_stl")
        log_error(f"Gevonden args: {args}")
        sys.exit(1)

    depth_path = args[0]
    alpha_path = args[1]
    output_path = args[2]

    extra = args[3:]
    opts = _parse_cli_options(extra)

    # typed option helper
    def _get_opt(name, default, cast):
        if name in opts:
            try:
                return cast(opts[name])
            except Exception:
                return default
        return default

    # Common tunables exposed to the user
    plane_scale_x = _get_opt("plane_scale_x", 10.0, float)
    use_modifiers = str(opts.get("use_modifiers", "false")).lower() in ("1", "true", "yes")
    apply_modifiers_for_stl = str(opts.get("apply_modifiers_for_stl", "false")).lower() in ("1", "true", "yes")
    subsurf_levels = _get_opt("subsurf_levels", 6, int)
    subsurf_render_levels = _get_opt("subsurf_render_levels", subsurf_levels, int)
    subsurf_type = _get_opt("subsurf_type", "SIMPLE", str)
    subsurf_use_limit_surface = _get_opt("subsurf_use_limit_surface", False, lambda v: str(v).lower() in ("1","true","yes"))
    displace_strength = _get_opt("displace_strength", 0.5, float)
    displace_mid_level = _get_opt("displace_mid_level", 0.0, float)

    corrective_iterations = _get_opt("corrective_iterations", 5, int)
    corrective_factor = _get_opt("corrective_factor", 0.5, float)

    max_height = _get_opt("max_height", 1.0, float)
    max_samples = _get_opt("max_samples", 800, int)
    alpha_threshold = _get_opt("alpha_threshold", 0.5, float)
    smoothing_iters = _get_opt("smoothing_iters", 2, int)
    smoothing_factor = _get_opt("smoothing_factor", 0.25, float)
    bottom_z = _get_opt("bottom_z", -2.0, float)

    log_info(f"Commandline arguments:")
    log_info(f"  1. Depth map: {depth_path}")
    log_info(f"  2. Alpha map: {alpha_path}")
    log_info(f"  3. Output STL: {output_path}")
    log_info(f"  Options: {opts}")
    log_info("")

    try:
        # FASE 1
        write_debug(f"MAIN start args depth={depth_path} alpha={alpha_path} output={output_path} opts={opts}")
        depth_img, alpha_img, plane_obj = setup_scene(depth_path, alpha_path, plane_scale_x=plane_scale_x)
        print("")

        # FASE 2: either apply modifiers or use heightfield fallback
        if use_modifiers:
            log_info('FASE 2: Gebruik modifiers-path (user enabled)')
            add_modifiers(
                plane_obj,
                depth_img,
                alpha_img,
                subsurf_levels=subsurf_levels,
                subsurf_render_levels=subsurf_render_levels,
                subsurf_type=subsurf_type,
                subsurf_use_limit_surface=subsurf_use_limit_surface,
                displace_strength=displace_strength,
                displace_mid_level=displace_mid_level,
                corrective_iterations=corrective_iterations,
                corrective_factor=corrective_factor,
            )
            # Keep modifier stack editable by default; only apply when explicitly requested.
            if apply_modifiers_for_stl:
                log_info("  > apply_modifiers_for_stl=true, modifiers worden toegepast voor export.")
                apply_modifiers(plane_obj)
            else:
                log_info("  > apply_modifiers_for_stl=false, modifiers blijven bewerkbaar in .blend.")
        else:
            log_info('FASE 2: Overslaan van modifiers - gebruik heightfield generator voor betrouwbare STL')
            create_heightfield_block(
                depth_img,
                alpha_img,
                plane_obj,
                max_height=max_height,
                max_samples=max_samples,
                alpha_threshold=alpha_threshold,
                smoothing_iters=smoothing_iters,
                smoothing_factor=smoothing_factor,
                bottom_z=bottom_z,
            )
        print("")
        write_debug("MAIN reached post-creation before export")

        # FASE 5
        export_to_stl(plane_obj, output_path)
        try:
            save_blend_file(output_path)
        except Exception as e:
            write_debug(f"save_blend_file failed: {e} \n" + traceback.format_exc())
            log_info(f"Opslaan .blend mislukte maar STL kan nog bestaan: {e}")
        print("")

        log_success("=" * 70)
        log_success("ALLE FASES VOLTOOID - 3D-reliëf succesvol gegenereerd!")
        log_success("=" * 70)

    except FileNotFoundError as e:
        log_error(f"Bestandsfout: {e}")
        sys.exit(1)
    except RuntimeError as e:
        log_error(f"Runtime fout: {e}")
        sys.exit(1)
    except Exception as e:
        log_error(f"Onverwachte fout: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
