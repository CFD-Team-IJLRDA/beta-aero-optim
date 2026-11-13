#!/usr/bin/env bash
set -euo pipefail

rm -rf constant/polyMesh constant/C*

gmsh cascade_grid_for_opt2.geo -3 -format msh2 -o mesh.msh # Change name of .geo file here if needed
gmshToFoam mesh.msh

# Modify boundary file directly via Python:
#   front  -> right  (empty)
#   back   -> left   (empty)
#   top    -> cyclicAMI (neighbourPatch bottom, separationVector (0 +0.04039 0))
#   bottom -> cyclicAMI (neighbourPatch top,    separationVector (0 -0.04039 0))
# NOTE: createPatch -overwrite in v2306 does NOT change the patch type when using
#       constructFrom patches, so we bypass it entirely and edit the boundary directly.
python3 - << 'PYEOF'
import re, sys

with open('constant/polyMesh/boundary', 'r') as f:
    content = f.read()

def modify_patch(content, old_name, new_name, new_entries):
    """Find the patch block for old_name and rewrite it with new_name/entries,
    preserving the original nFaces and startFace values."""
    pat = (r'([ \t]+)' + re.escape(old_name) + r'[ \t]*\n'
           r'([ \t]+\{)([^}]+)(\})')
    def repl(m):
        indent, ob, body, cb = m.group(1), m.group(2), m.group(3), m.group(4)
        nf_m = re.search(r'nFaces\s+\d+;',    body)
        sf_m = re.search(r'startFace\s+\d+;', body)
        if not nf_m or not sf_m:
            print(f'ERROR: could not find nFaces/startFace in block for {old_name}', file=sys.stderr)
            sys.exit(1)
        ei = '        '
        new_body = '\n' + ''.join(ei + e + '\n' for e in new_entries)
        new_body += ei + nf_m.group(0) + '\n'
        new_body += ei + sf_m.group(0) + '\n'
        new_body += indent
        return indent + new_name + '\n' + ob + new_body + cb
    result = re.sub(pat, repl, content, flags=re.DOTALL)
    if result == content:
        print(f'WARNING: patch "{old_name}" not found in boundary file', file=sys.stderr)
    return result

content = modify_patch(content, 'front', 'right', [
    'type            empty;',
    'physicalType    empty;',
])
content = modify_patch(content, 'back', 'left', [
    'type            empty;',
    'physicalType    empty;',
])
content = modify_patch(content, 'top', 'top', [
    'type            cyclicAMI;',
    'physicalType    cyclicAMI;',
    'inGroups        1(cyclicAMI);',
    'neighbourPatch  bottom;',
    'transform       translational;',
    'separationVector (0 0.04039 0);',
    'matchTolerance  0.0001;',
])
content = modify_patch(content, 'bottom', 'bottom', [
    'type            cyclicAMI;',
    'physicalType    cyclicAMI;',
    'inGroups        1(cyclicAMI);',
    'neighbourPatch  top;',
    'transform       translational;',
    'separationVector (0 -0.04039 0);',
    'matchTolerance  0.0001;',
])
content = modify_patch(content, 'airfoil', 'airfoil', [
    'type            wall;',
    'physicalType    wall;',
    'inGroups        1(wall);',
])

with open('constant/polyMesh/boundary', 'w') as f:
    f.write(content)
print('Boundary file patched successfully.')
PYEOF

checkMesh
rm -rf 0
cp -r 0.org 0
# Regenerate cell-centre fields (C, Cx, Cy, Cz) for the current mesh.
postProcess -func writeCellCentres -time 0
