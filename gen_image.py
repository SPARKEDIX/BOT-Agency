from agency.image_maker import ImageMakerAgent
out = ImageMakerAgent().run("a small red robot toy, studio photo, high detail")
print(out["output"])
print("FILE:" + out["file"])
