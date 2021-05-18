import bpy
import mathutils

scene = bpy.context.scene

cam_data = bpy.data.cameras.new('camera')
#name your camera
cam = bpy.data.objects.new('camera', cam_data)
scene.objects.link(cam)
scene.camera = cam

#Name your camera's location
#cam.location = mathutils.Vector((6, -3, 5))
#cam.rotation_euler = mathutils.Euler((0.9, 0.0, 1.1))

scene.render.image_settings.file_format = 'PNG'
scene.render.filepath = "F:/image.png"
#change filepath to your system's
bpy.ops.render.render(write_still = 1)
