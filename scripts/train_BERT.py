from data_processing import pipeline
from model import prepare_datasets, create_QA_model

train_dataset = pipeline('train')
val_dataset = pipeline('validation')
test_dataset = pipeline('test')

train_set = prepare_datasets(train_dataset)
validation_set = prepare_datasets(val_dataset)

model = create_QA_model()

model.fit(train_set, validation_data=validation_set, epochs=3)
model.save_pretrained('DistilBERT/')