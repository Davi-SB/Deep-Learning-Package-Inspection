import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Concatenate
from tensorflow.keras.applications import VGG16, EfficientNetB0
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

# --- 1. CONFIGURAÇÕES GERAIS ---
TRAIN_DIR = os.path.join("images", "removed_bg", "proporcional_resize")
INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.0005
VALIDATION_SPLIT = 0.2

if not os.path.isdir(TRAIN_DIR):
    raise FileNotFoundError(f"ERRO: O diretorio de treino '{TRAIN_DIR}' nao foi encontrado.")
print(f"Diretorio de treino definido como: '{TRAIN_DIR}'")

# --- 2. GERADOR DE DADOS PERSONALIZADO ---
class DualInputGenerator(tf.keras.utils.Sequence):
    def __init__(self, directory, batch_size, target_size, preprocess_side, preprocess_top, validation_split=0.0, subset=None, seed=123):
        super().__init__()
        self.directory = directory
        self.batch_size = batch_size
        self.target_size = target_size
        self.preprocess_side = preprocess_side
        self.preprocess_top = preprocess_top
        self.class_indices = {'damaged': 0, 'intact': 1}
        self.class_names = list(self.class_indices.keys())
        all_pairs, all_labels = self._scan_image_pairs()
        if not all_pairs:
            raise ValueError("Nenhum par de imagens foi encontrado. Verifique a estrutura de pastas.")
        
        indices = np.arange(len(all_pairs))
        np.random.seed(seed)
        np.random.shuffle(indices)
        
        if validation_split > 0 and subset:
            split_point = int(len(all_pairs) * (1 - validation_split))
            if subset == 'training':
                self.image_pairs = [all_pairs[i] for i in indices[:split_point]]
                self.labels = [all_labels[i] for i in indices[:split_point]]
            elif subset == 'validation':
                self.image_pairs = [all_pairs[i] for i in indices[split_point:]]
                self.labels = [all_labels[i] for i in indices[split_point:]]
            else:
                raise ValueError("O 'subset' deve ser 'training' ou 'validation'.")
        else:
            self.image_pairs = all_pairs
            self.labels = all_labels
        self.samples = len(self.image_pairs)

    def _scan_image_pairs(self):
        pairs = []
        labels = []
        print("Buscando pares de imagens 'side' e 'top'...")
        for class_name, class_idx in self.class_indices.items():
            side_dir = os.path.join(self.directory, 'side', class_name)
            top_dir = os.path.join(self.directory, 'top', class_name)
            if not os.path.isdir(side_dir) or not os.path.isdir(top_dir):
                print(f"Aviso: Diretórios para a classe '{class_name}' não encontrados. Pulando.")
                continue
            side_files = sorted(os.listdir(side_dir))
            top_files = sorted(os.listdir(top_dir))
            for side_file, top_file in zip(side_files, top_files):
                pairs.append((os.path.join(side_dir, side_file), os.path.join(top_dir, top_file)))
                labels.append(class_idx)
        print(f"Encontrados {len(pairs)} pares de imagens.")
        return pairs, labels

    def __len__(self):
        return int(np.ceil(self.samples / self.batch_size))

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size
        batch_pairs = self.image_pairs[start_idx:end_idx]
        batch_labels = self.labels[start_idx:end_idx]
        
        x_side_batch = np.zeros((len(batch_pairs), *self.target_size, 3), dtype='float32')
        x_top_batch = np.zeros((len(batch_pairs), *self.target_size, 3), dtype='float32')
        y_batch = np.array(batch_labels, dtype='uint8')
        
        for i, (side_path, top_path) in enumerate(batch_pairs):
            img_side = load_img(side_path, target_size=self.target_size)
            x_side = img_to_array(img_side)
            x_side_batch[i] = self.preprocess_side(x_side)
            
            img_top = load_img(top_path, target_size=self.target_size)
            x_top = img_to_array(img_top)
            x_top_batch[i] = self.preprocess_top(x_top)
            
        return {'side_input': x_side_batch, 'top_input': x_top_batch}, y_batch

# --- 3. CRIAÇÃO DOS GERADORES ---
train_generator = DualInputGenerator(
    TRAIN_DIR, BATCH_SIZE, INPUT_SHAPE[:2],
    preprocess_side=vgg16_preprocess,
    preprocess_top=effnet_preprocess,
    validation_split=VALIDATION_SPLIT,
    subset='training'
)

validation_generator = DualInputGenerator(
    TRAIN_DIR, BATCH_SIZE, INPUT_SHAPE[:2],
    preprocess_side=vgg16_preprocess,
    preprocess_top=effnet_preprocess,
    validation_split=VALIDATION_SPLIT,
    subset='validation'
)

# --- 4. DEFINIÇÃO DO MODELO ---
def create_dual_model(input_shape):
    base_side = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    base_side.trainable = False
    
    base_top = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    base_top.trainable = False
    
    input_side = Input(shape=input_shape, name='side_input')
    input_top = Input(shape=input_shape, name='top_input')
    
    features_side = base_side(input_side)
    features_top = base_top(input_top)
    
    flat_side = Flatten()(features_side)
    flat_top = Flatten()(features_top)
    
    concatenated = Concatenate()([flat_side, flat_top])
    
    x = Dense(512, activation='relu')(concatenated)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)
    
    return Model(inputs=[input_side, input_top], outputs=output)

# --- 5. COMPILAÇÃO E TREINAMENTO ---
dual_model = create_dual_model(INPUT_SHAPE)
dual_model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='binary_crossentropy', metrics=['accuracy'])
dual_model.summary()

print("\nIniciando o treinamento do modelo...")
history = dual_model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)
    ]
)

# --- 6. MATRIZ DE CONFUSÃO (MÉTODO CORRIGIDO) ---
print("\nGerando a Matriz de Confusão para o conjunto de validação...")

# 1. Obter as previsões para TODO o conjunto de dados de uma só vez.
# Passe o gerador inteiro para o método predict. É mais eficiente e evita erros.
y_pred_probs = dual_model.predict(validation_generator, verbose=1)

# 2. Obter os rótulos verdadeiros diretamente do gerador.
# O atributo .labels que criamos na classe armazena todos os rótulos na ordem correta.
y_true = validation_generator.labels

# 3. Garanta que o número de predições e rótulos seja o mesmo.
# A predição pode retornar menos itens se o tamanho do dataset não for divisível pelo batch_size.
# Por isso, ajustamos y_true para ter o mesmo tamanho das predições.
if len(y_pred_probs) != len(y_true):
    print("Aviso: Ajustando o número de rótulos para corresponder às predições.")
    y_true = y_true[:len(y_pred_probs)]

# 4. Converter as probabilidades em classes (0 ou 1)
y_pred_classes = (y_pred_probs.flatten() > 0.5).astype(int)

# 5. Calcular a matriz de confusão
cm = confusion_matrix(y_true, y_pred_classes)
class_names = validation_generator.class_names

# 6. Plotar a matriz de confusão de forma visual
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Matriz de Confusão', fontsize=16)
plt.ylabel('Classe Verdadeira', fontsize=12)
plt.xlabel('Classe Prevista', fontsize=12)
plt.show()


# --- 7. GRÁFICOS DE DESEMPENHO DO TREINAMENTO ---
print("\nGerando gráficos de desempenho do treinamento...")
plt.figure(figsize=(14, 6))

# Gráfico da Perda (Loss)
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Perda de Treino')
plt.plot(history.history['val_loss'], label='Perda de Validação')
plt.title('Evolução da Perda (Loss)')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()

# Gráfico da Acurácia
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Acurácia de Treino')
plt.plot(history.history['val_accuracy'], label='Acurácia de Validação')
plt.title('Evolução da Acurácia')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()

plt.tight_layout()
plt.show()

# --- 8. INTERPRETABILIDADE (FUNÇÃO CORRIGIDA) ---
def integrated_gradients(input_data, model, input_name, baseline=None, steps=50):
    if baseline is None:
        baseline = np.zeros_like(input_data).astype(np.float32)

    interpolated_path = [baseline + (float(i) / steps) * (input_data - baseline) for i in range(steps + 1)]
    interpolated_inputs_np = np.concatenate(interpolated_path, axis=0)

    other_input_name = 'top_input' if input_name == 'side_input' else 'side_input'
    feed_dict_np = {
        input_name: interpolated_inputs_np,
        other_input_name: np.zeros_like(interpolated_inputs_np)
    }

    feed_dict_tf = {key: tf.convert_to_tensor(val, dtype=tf.float32) for key, val in feed_dict_np.items()}

    with tf.GradientTape() as tape:
        tape.watch(feed_dict_tf[input_name])
        preds = model(feed_dict_tf)
        outputs = preds[:, 0]

    grads = tape.gradient(outputs, feed_dict_tf[input_name])
    avg_grads = np.mean(grads.numpy().reshape((steps + 1, *input_data.shape)), axis=0)
    integrated_grads = (input_data - baseline) * avg_grads
    return np.sum(integrated_grads, axis=-1)[0]

def display_integrated_gradients(img_path, preprocess_fn, input_name, model):
    if not os.path.exists(img_path):
        print(f"Imagem de exemplo não encontrada: {img_path}")
        return
        
    print(f"\nGerando mapa de atribuição para: {os.path.basename(img_path)}")
    img = load_img(img_path, target_size=INPUT_SHAPE[:2])
    arr = img_to_array(img)
    input_arr = preprocess_fn(arr)
    input_arr = np.expand_dims(input_arr, axis=0)
    
    attribution = integrated_gradients(input_arr, model, input_name=input_name)
    
    # Normaliza a atribuição para uma melhor visualização
    attribution = (attribution - np.min(attribution)) / (np.max(attribution) - np.min(attribution) + 1e-8)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.imshow(attribution, cmap='hot', alpha=0.5)
    plt.title(f'Integrated Gradients: {input_name}')
    plt.axis('off')
    plt.show()

def run_integrated_gradients_example():
    print("\n--- Iniciando Análise de Interpretabilidade ---")
    side_img_path = os.path.join(TRAIN_DIR, "side", "damaged", "0185507921789_side.png")
    top_img_path = os.path.join(TRAIN_DIR, "top", "damaged", "0185507921789_top.png")
    
    if not os.path.exists(side_img_path) or not os.path.exists(top_img_path):
        print("AVISO: Imagens de exemplo para interpretabilidade não encontradas. Pulando esta etapa.")
        return
        
    display_integrated_gradients(side_img_path, vgg16_preprocess, 'side_input', dual_model)
    display_integrated_gradients(top_img_path, effnet_preprocess, 'top_input', dual_model)

# --- CHAMADA FINAL ---
run_integrated_gradients_example()

print("\nAnálise completa.")
