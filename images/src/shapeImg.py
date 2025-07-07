from PIL import Image
import os

def simple_resize_with_black_bg(input_path: str, output_path: str, size: tuple = (244, 244)):
    try:
        # Abre a imagem original
        original_img = Image.open(input_path)

        # Cria uma nova imagem com fundo preto
        # Isso garante que qualquer transparência seja substituída por preto
        black_background = Image.new("RGB", original_img.size, (0, 0, 0))
        
        # Cola a imagem original sobre o fundo preto.
        # Se a imagem for RGBA, a transparência é usada como máscara. Se for RGB, ela só é copiada.
        black_background.paste(original_img, mask=original_img.getchannel('A') if original_img.mode == 'RGBA' else None)

        # Redimensiona a imagem para o tamanho exato (pode distorcer)
        # Usamos Image.Resampling.LANCZOS para a melhor qualidade de redução
        resized_img = black_background.resize(size, Image.Resampling.LANCZOS)

        # Salva a imagem resultante
        resized_img.save(output_path)
        print(f"Imagem redimensionada (simples) e salva em: {output_path}")

    except FileNotFoundError:
        print(f"Erro: O arquivo de entrada não foi encontrado em '{input_path}'")
        input()
    except Exception as e:
        print(f"Ocorreu um erro: {e}")
        input()

def proportional_resize_with_padding(input_path: str, output_path: str, canvas_size: tuple = (244, 244)):
    try:
        # Abre a imagem original
        original_img = Image.open(input_path)

        # Garante que a imagem esteja em modo RGB, tratando a transparência
        rgb_img = Image.new("RGB", original_img.size, (0, 0, 0))
        rgb_img.paste(original_img, mask=original_img.getchannel('A') if original_img.mode == 'RGBA' else None)

        # Calcula a proporção para o novo tamanho sem distorcer
        ratio = min((canvas_size[0] / rgb_img.width), (canvas_size[1] / rgb_img.height))
        new_width = int(rgb_img.width * ratio)
        new_height = int(rgb_img.height * ratio)

        # Redimensiona a imagem mantendo a proporção
        resized_img = rgb_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Cria o canvas final com fundo preto
        black_canvas = Image.new('RGB', canvas_size, (0, 0, 0))

        # Calcula a posição para colar a imagem redimensionada no centro do canvas
        paste_x = (canvas_size[0] - new_width) // 2
        paste_y = (canvas_size[1] - new_height) // 2

        # Cola a imagem redimensionada no canvas
        black_canvas.paste(resized_img, (paste_x, paste_y))

        # Salva a imagem final
        black_canvas.save(output_path)
        print(f"Imagem redimensionada (proporcional) e salva em: {output_path}")

    except FileNotFoundError:
        print(f"Erro: O arquivo de entrada não foi encontrado em '{input_path}'")
        input()
    except Exception as e:
        print(f"Ocorreu um erro: {e}")
        input()

if __name__ == '__main__':    
    processing_jobs = [
        {
            "job_name":            r"crop_only - side - damaged",
            "input_dir":           r"images\crop_only\original_shape\side\damaged",
            "output_simple":       r"images\crop_only\simple_resize\side\damaged",
            "output_proportional": r"images\crop_only\proporcional_resize\side\damaged"
        },
        {
            "job_name":            r"crop_only - side - intact",
            "input_dir":           r"images\crop_only\original_shape\side\intact",
            "output_simple":       r"images\crop_only\simple_resize\side\intact",
            "output_proportional": r"images\crop_only\proporcional_resize\side\intact"
        },
        
        {
            "job_name":            r"crop_only - top - damaged",
            "input_dir":           r"images\crop_only\original_shape\top\damaged",
            "output_simple":       r"images\crop_only\simple_resize\top\damaged",
            "output_proportional": r"images\crop_only\proporcional_resize\top\damaged"
        },
        {
            "job_name":            r"crop_only - top - intact",
            "input_dir":           r"images\crop_only\original_shape\top\intact",
            "output_simple":       r"images\crop_only\simple_resize\top\intact",
            "output_proportional": r"images\crop_only\proporcional_resize\top\intact"
        },
        
        ########
        
        {
            "job_name":            r"removed_bg - side - damaged",
            "input_dir":           r"images\removed_bg\original_shape\side\damaged",
            "output_simple":       r"images\removed_bg\simple_resize\side\damaged",
            "output_proportional": r"images\removed_bg\proporcional_resize\side\damaged"
        },
        
        {
            "job_name":            r"removed_bg - side - intact",
            "input_dir":           r"images\removed_bg\original_shape\side\intact",
            "output_simple":       r"images\removed_bg\simple_resize\side\intact",
            "output_proportional": r"images\removed_bg\proporcional_resize\side\intact"
        },
        
        {
            "job_name":            r"removed_bg - top - damaged",
            "input_dir":           r"images\removed_bg\original_shape\top\damaged",
            "output_simple":       r"images\removed_bg\simple_resize\top\damaged",
            "output_proportional": r"images\removed_bg\proporcional_resize\top\damaged"
        },
        
        {
            "job_name":            r"removed_bg - top - intact",
            "input_dir":           r"images\removed_bg\original_shape\top\intact",
            "output_simple":       r"images\removed_bg\simple_resize\top\intact",
            "output_proportional": r"images\removed_bg\proporcional_resize\top\intact"
        },
    ]
    
    # Loop através de cada tarefa de processamento
    for job in processing_jobs:
        print(f"\n--- Iniciando Tarefa: {job['job_name']} ---")
        
        # Cria os diretórios de saída para a tarefa atual
        os.makedirs(job['output_simple'], exist_ok=True)
        os.makedirs(job['output_proportional'], exist_ok=True)
        print(f"  -> Saída (simples):      {job['output_simple']}")
        print(f"  -> Saída (proporcional): {job['output_proportional']}")

        # Loop através de cada diretório de entrada da tarefa
        input_dir = job['input_dir']
        if not os.path.isdir(input_dir):
            print(f"  [AVISO] Diretório de entrada não encontrado, pulando: {input_dir}")
            input()
            continue
        
        print(f"  Lendo de: {input_dir}")
        
        # Loop através de cada arquivo no diretório de entrada
        for filename in os.listdir(input_dir):
            if filename.lower().endswith('.png'):
                input_path = os.path.join(input_dir, filename)
                
                # Define os caminhos de saída
                simple_output_path = os.path.join(job['output_simple'], filename)
                proportional_output_path = os.path.join(job['output_proportional'], filename)
                
                # Executa ambas as funções
                simple_resize_with_black_bg(input_path, simple_output_path)
                proportional_resize_with_padding(input_path, proportional_output_path)
        
        print(f"  Diretório '{input_dir}' concluído.")

    print("\n--- Processamento em Concluído ---")