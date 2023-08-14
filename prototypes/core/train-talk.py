from transformers import GPT2LMHeadModel, GPT2Tokenizer
import random

def train_model_from_pdf(pdf_path):
    # Tu código para entrenar el modelo a partir del PDF
    # ...
    # trainer = Trainer()
    # trainer.train(pdf_path)
    print("No train available!")

def save_model_to_disk(output_dir):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

def load_model_from_disk(model_dir):
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    return model, tokenizer

def main():
    while True:
        print("Menú:")
        print("a) Entrenar")
        print("b) Conversar")
        print("c) Salir")
        choice = input("Elige una opción: ")

        if choice == "a":
            pdf_path = input("Introduce la ruta del PDF para el entrenamiento: ")  # 'base.pdf'
            train_model_from_pdf(pdf_path)
            model_output_dir = input("Introduce la ruta para guardar los modelos entrenados: ") #  'presets/' + pdf_path 
            save_model_to_disk(model_output_dir)
            print("Modelos guardados exitosamente en: " + model_output_dir)

        elif choice == "b":
            model_dir =  input("Introduce la ruta del directorio con los modelos entrenados: ")  # 'presets/base.pdf'
            model, tokenizer = load_model_from_disk(model_dir)

            while True:
                user_input = input("Usuario: ")
                input_ids = tokenizer.encode(user_input, return_tensors="pt")
                output = model.generate(input_ids, max_length=100, num_return_sequences=1)
                generated_response = tokenizer.decode(output[0], skip_special_tokens=True)
                print("Modelo:", generated_response)

                continue_conversation = input("¿Quieres continuar conversando? (s/n): ")
                if continue_conversation.lower() != "s":
                    break

        elif choice == "c":
            break

        else:
            print("Opción inválida. Por favor, elige 'a', 'b' o 'c'.")

if __name__ == "__main__":
    main()
