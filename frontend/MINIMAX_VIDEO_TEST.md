# MiniMax image-to-video API test

Start the frontend, then replace `PASTE_BASE64_IMAGE_HERE` with the base64 image data (without a second `data:image/...;base64,` prefix):

```bash
curl -X POST http://localhost:3000/api/minimax-video \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Preserve the exact photorealistic Martian terrain composition, orange-red palette, rocky foreground, valleys, distant mountain ranges, atmospheric depth, and warm low sunlight from the reference image. Create a professional cinematic establishing shot from a low rover-mounted camera. The camera glides slowly forward across the surface with a very subtle right-to-left lateral drift, creating natural parallax between foreground rocks, middle-ground terrain, and distant mountains. Motion is smooth, stable, slow, and physically realistic. The landscape and every rock remain completely stable. Only the camera moves. Keep haze subtle and confined to the far distance. No people, robots, buildings, spacecraft, text, logos, UI, blowing sand, rising sand, dust clouds, dust storms, floating particles, smoke, foreground fog, debris, moving rocks, terrain deformation, melting, warping, rippling, flickering, camera shake, sudden turns, zoom jumps, rapid acceleration, animated sun, animated clouds, neon colors, or fantasy elements.",
    "duration": 6,
    "resolution": "1080P",
    "firstFrameImage": "data:image/jpeg;base64,PASTE_BASE64_IMAGE_HERE"
  }'
```

Check the returned task ID:

```bash
curl "http://localhost:3000/api/minimax-video?taskId=PASTE_TASK_ID_HERE"
```
