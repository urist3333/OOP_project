from MnistDataLoader import MnistDataLoader


my_data_loader = MnistDataLoader("mnist_bw")
tr_data = my_data_loader.get_training_data()
te_data = my_data_loader.get_testing_data()
labels = my_data_loader.get_labels()

my_data_loader = MnistDataLoader("mnist_color",version = "m1")
tr_data = my_data_loader.get_training_data()
te_data = my_data_loader.get_testing_data()
labels = my_data_loader.get_labels()

print(f"shape labels{labels.shape}")
print(f"shape te_data {te_data.shape}")


print(f"shape labels{labels.shape}")
print(f"shape te_data {te_data.shape}")

my_data_loader = MnistDataLoader("mnist_color",version ="m1")
tr_data = my_data_loader.get_training_data()
te_data = my_data_loader.get_testing_data()
labels = my_data_loader.get_labels()

print(f"shape labels{labels.shape}")
print(f"shape te_data {te_data.shape}")


my_data_loader = MnistDataLoader("mnist_bw")
tr_data = my_data_loader.get_training_data()
te_data = my_data_loader.get_testing_data()
labels = my_data_loader.get_labels()

print(f"shape labels{labels.shape}")
print(f"shape te_data {te_data.shape}")

