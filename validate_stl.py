import sys
import trimesh

if len(sys.argv) < 2:
    print('Usage: validate_stl.py <path.stl>')
    sys.exit(2)

p = sys.argv[1]
print('Validating', p)
try:
    m = trimesh.load(p, force='mesh')
    print('vertices', len(m.vertices))
    print('faces', len(m.faces))
    print('is_watertight', m.is_watertight)
    print('is_winding_consistent', m.is_winding_consistent)
    print('euler_number', m.euler_number)
    print('bounds', m.bounds)
    if not m.is_watertight:
        print('Mesh is not watertight — attempting repair...')
        m.repair.fill_holes()
        m.remove_degenerate_faces()
        m.merge_vertices()
        m.export(p)
        print('Repaired and exported back to', p)
except Exception as e:
    print('Error loading STL:', e)
    sys.exit(1)
