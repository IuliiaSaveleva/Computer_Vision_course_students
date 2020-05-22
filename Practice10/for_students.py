import nmslib # pip install nmslib

from tensorboardX import SummaryWriter

from dataset_class import ValidationDataset



class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


# функция валидации, пример вызова смотри в конце файла
def validate_model(query_labels, query_embeddings, retrieval_labels, retrieval_embeddings):
    index = nmslib.init(method='hnsw', space='l2')
    index.addDataPointBatch(retrieval_embeddings)
    index.createIndex(print_progress=True)
    search_count = [1, 5, 10, 20]
    matches = np.zeros((len(search_count), len(query_labels)))
    for i in range(len(query_labels)):
        quary_label = query_labels[i]
        ids, distances = index.knnQuery(query_embeddings[i], k=max(search_count))
        local_retrieval_labels = retrieval_labels[ids]
        for j in range(len(search_count)):
            matches[j][i] = any(quary_label == local_retrieval_labels[:search_count[j]])
    for i in range(len(search_count)):
        print('Top', str(search_count[i]), '-', str(np.sum(matches[i]) / float(len(query_labels)) * 100.) + '%')




validation_query_and_retrieval_pickle = 'validation_dataset.pickle'
# Кастомный датасет, который считает картинки из пикла
# Путь до файла в пикле выглядит так:
# 2_151310032613/chair_final/chair_final_151310032613_0.JPG
# categoryId_productId/categoryName/categoryName_productId_imageNumber.JPG
# необходимо подстроить под свою структуру датасета пути в пикле и
# формирование путей до картинок в ValidationDataset (функции _find_classes и make_dataset)
# не забудь, что классы - это продукты, а не категории :)
valid_dataset = ValidationDataset(root=valid_dir,
                                     query_and_retrieval_pickle=validation_query_and_retrieval_pickle,
                                     transform=data_valid_transform)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32,
                                               shuffle=False, num_workers=1)

# чтобы картинки в Tensorboard Projector были адекватными
unnormalize = UnNormalize(mean=mean, std=std)

model.train(False)
images = []
labels = []
embeddings = []
for data in valid_loader:
    inputs_batch, labels_batch = data
    inputs_batch = Variable(inputs_batch).to(device)
    outputs = model(inputs_batch)
    outputs = F.normalize(outputs, p=2, dim=1)
    for i in range(len(inputs_batch)):
        images.append(unnormalize(inputs_batch[i]).cpu().detach().numpy())
        labels.append(labels_batch[i])
        embeddings.append(outputs[i].cpu().detach().numpy())

images = np.array(images)
embeddings = np.array(embeddings)
labels = np.array(labels)

# добавляем 1000 случайных эмбедингов для визуализации в Tensorboard Projector
# обязательно реализовать у себя визуализацию эмбедингов, так как я помогаю
# с реализацией, то за невыполнение будет снижаться балл!
writer = SummaryWriter('logs/') # можно писать и в ту же папку логов, что и при обучении
random_indices = np.arange(len(embeddings))
np.random.shuffle(random_indices)
random_indices = random_indices[:1000]
projector_embeddings = embeddings[random_indices]
projector_images = images[random_indices]
projector_class_labels = [valid_dataset.classes[label] for label in labels[random_indices]]
writer.add_embedding(projector_embeddings,
                     metadata=projector_class_labels,
                     label_img=projector_images, global_step=iter)

# первые query_count элементов датасета - query картинки, остальные retrieval set
query_labels = labels[:valid_dataset.query_count]
query_embeddings = embeddings[:valid_dataset.query_count]
retrieval_labels = labels[valid_dataset.query_count:]
retrieval_embeddings = embeddings[valid_dataset.query_count:]
validate_model(query_labels, query_embeddings, retrieval_labels, retrieval_embeddings)
