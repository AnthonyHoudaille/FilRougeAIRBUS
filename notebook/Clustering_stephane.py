new_model = clf_model
new_model.layers.pop()

dataframe_gen = make_image_gen(data_link_unbalanced, train_image_dir, 2, (3, 3))
pred = new_model.predict_generator(dataframe_gen, steps=len(dataframe)//2, verbose=1)

pred.shape # (231722, 1)

new_model.layers.pop()
new_model.summary()

pred2 = new_model.predict_generator(dataframe_gen, steps=len(dataframe)//2, verbose=1)
pred2.shape #(231722, 1)