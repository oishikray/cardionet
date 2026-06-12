AHA LV Bullseye / Segment Assignment Notes

This project uses the AHA left-ventricular myocardial segmentation convention for short-axis cardiac MRI. The key point is that the AHA model is not just “split the myocardium into equal angular bins.” The bins must be anatomically anchored.

The LV myocardium is divided along two axes:

First, along the long axis of the ventricle, from base to apex:

Basal ring: segments 1–6
Mid-cavity ring: segments 7–12
Apical ring: segments 13–16

Second, circumferentially on short-axis slices:

Basal and mid-cavity slices are divided into six 60° sectors each.
Apical slices are divided into four 90° sectors.

The critical anatomical anchor is the RV insertion/contact with the LV myocardium. The attachment of the right ventricular wall to the LV is what identifies the septal side. This must be used to separate the septum from the LV free wall.

Do not assume fixed image orientation such as “top of image is anterior” unless this has already been explicitly standardized by preprocessing. The sector placement should be derived from the segmentation geometry where possible.

For basal and mid-cavity rings, the six sectors are ordered anatomically as:

anterior
anteroseptal
inferoseptal
inferior
inferolateral
anterolateral

For basal slices, these correspond to:

Segment 1 = basal anterior
Segment 2 = basal anteroseptal
Segment 3 = basal inferoseptal
Segment 4 = basal inferior
Segment 5 = basal inferolateral
Segment 6 = basal anterolateral

For mid-cavity slices, the same circumferential order is used:

Segment 7 = mid anterior
Segment 8 = mid anteroseptal
Segment 9 = mid inferoseptal
Segment 10 = mid inferior
Segment 11 = mid inferolateral
Segment 12 = mid anterolateral

For apical slices, the LV tapers, so there are only four sectors:

Segment 13 = apical anterior
Segment 14 = apical septal
Segment 15 = apical inferior
Segment 16 = apical lateral

Important implementation detail: basal and mid-cavity rings do not use four quadrants. They use six equal 60° sectors. Only the apical ring uses four 90° sectors.

The bullseye display is a flattened anatomical summary:

Outer ring: basal segments 1–6
Middle ring: mid-cavity segments 7–12
Inner ring: apical segments 13–16

The visual ordering should match the AHA convention:

anterior at the top
inferior at the bottom
septal / anteroseptal / inferoseptal on the side determined by RV attachment
lateral / anterolateral / inferolateral on the opposite free-wall side

The algorithm should therefore do roughly this:

For each short-axis slice/frame, identify LV cavity, myocardium, and RV masks.
Compute the LV centroid from the LV cavity or LV + myocardium geometry.
Find the RV–LV/myocardium contact region, or the nearest RV-facing myocardial boundary.
Use that RV contact/insertion direction to determine the septal side.
Define the anterior/inferior/septal/lateral coordinate frame from the RV insertion geometry, not from raw image rows/columns alone.
Assign angular bins around the LV centroid according to the AHA sector order.
For basal and mid slices, assign six 60° bins:
basal: 1, 2, 3, 4, 5, 6
mid: 7, 8, 9, 10, 11, 12
For apical slices, assign four 90° bins:
13, 14, 15, 16

Do not rotate labels arbitrarily to make a plot “look nice.” The labels must be tied to the anatomical orientation. The RV attachment defines the septal side; from there the AHA labels follow.

A useful sanity check is this: in basal and mid slices, the septal segments must be the two sectors adjacent to the RV attachment side: anteroseptal and inferoseptal. The opposite side should be the lateral free wall: anterolateral and inferolateral. If the RV is visually on the left/right/top/bottom because of preprocessing rotation, the labels should rotate with the anatomy, not stay fixed to image coordinates.

For the bullseye plot specifically, display using the standard AHA layout regardless of source image orientation.

Outer ring:

Segment 1 = basal anterior, at top
Segment 2 = basal anteroseptal
Segment 3 = basal inferoseptal
Segment 4 = basal inferior, at bottom
Segment 5 = basal inferolateral
Segment 6 = basal anterolateral

Middle ring:

Segment 7 = mid anterior, at top
Segment 8 = mid anteroseptal
Segment 9 = mid inferoseptal
Segment 10 = mid inferior, at bottom
Segment 11 = mid inferolateral
Segment 12 = mid anterolateral

Inner ring:

Segment 13 = apical anterior, at top
Segment 14 = apical septal
Segment 15 = apical inferior, at bottom
Segment 16 = apical lateral

The important distinction is:

Image-space sector assignment should be anatomically derived from the segmentation. Bullseye-space plotting should use the canonical AHA display layout.

The RV attachment does not directly mean “this exact ray is segment 2” or “this exact ray is segment 3.” It identifies the septal side and helps define the rotation of the LV coordinate frame. Once that frame is established, the AHA sectors are assigned in the standard circumferential order. In other words, RV contact is the anatomical compass, not itself a segment label.