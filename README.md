# curved_slits


This module extracts data along perpendicular slices from a guide line.



Basic usage:

```
from curved_slits import spline_slit, PointPicker
out = PointPicker(image)
points = out.return_points()

slits, _ = spline_slit(image[np.newaxis,], points)
``` 
