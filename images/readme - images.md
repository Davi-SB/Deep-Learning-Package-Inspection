# Organização dos diretórios de Imagens para Treinamento

Este documento descreve a estrutura de diretórios usados para o armazenamento do dataset de imagens do projeto. O objetivo é organizar os dados de forma lógica e padronizada, garantindo que estejam prontos para serem utilizados no treinamento dos modelos.

## Lógica da Estrutura

A organização segue uma lógica hierárquica de 4 níveis, que vai do tipo de processamento geral até o rótulo final de cada imagem.

1.  **Abordagem de Processamento (`crop_only`, `removed_bg`, ...)**
    O primeiro nível separa as imagens com base na principal técnica de pré-processamento aplicada.
    -   `crop_only/`: Contém imagens que passaram apenas por um processo de enquadramento nas caixas.
    -   `removed_bg/`: Contém imagens com as caixas enquadradas que tiveram seu fundo removido.

2.  **Formato da Imagem (`original_shape`, `proporcional_resize` e `simple_resize`)**
    Para cada abordagem, as imagens são salvas em três formatos distintos, oferecendo flexibilidade no uso dos dados.
    -   `original_shape/`: Imagens mantidas na sua resolução e proporção originais.
    -   `proporcional_resize/`: Imagens redimensionadas proporcionalmente para 224x224 mantendo sua proporção original, evitando distorções.
    -   `simple_resize/`: Imagens redimensionadas para o tamanho fixo de 224x224, podendo haver distorção.

3.  **Perspectiva da Imagem (`side` e `top`)**
    Dentro de cada formato, as imagens são agrupadas pela perspectiva da câmera.
    -   `side/`: Imagens com visão lateral do objeto.
    -   `top/`: Imagens com visão superior do objeto.

4.  **Rótulo para Treinamento (`intact` e `damaged`)**
    Cada pasta de perspectiva (`side` e `top`) contém duas subpastas que servem como **labels** para a classificação das imagens.
    -   `intact/`: Contém todas as imagens de objetos **intactos**.
    -   `damaged/`: Contém todas as imagens de objetos **danificados**.

### Exemplo de um Caminho Completo:

```plaintext
Abordagem (crop_only, removed_bg, ...)
└── Formato (original_shape, proporcional_resize ou simple_resize)
    └── Perspectiva (side ou top)
        ├── intact/
        │   ├── imagem_001.png
        │   └── imagem_002.png
        └── damaged/
            ├── imagem_101.png
            └── imagem_102.png