import support_functions as sf

model = sf.load_working_model('models/urte_3d_2')
images = sf.load_images('images')

#run cellpose model
channels = [[0,0]] #0, 0 means grayscale image
masks, flows, styles = model.eval(images, diameter=17, channels=channels, compute_masks=True)

#create Labels
labeled_images = sf.save_masks_overlay(images, masks, 'masks_new')

#Create 3d stack
sf.create_3d_stack(masks)
# sf.create_3d_projection(masks) Note very slow because of the size of the images.

number_of_cells = sf.count_objects(masks)



 
