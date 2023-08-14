import os
import pdfplumber
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import random

from nosce import Trainer

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
            pdf_path = 'target.pdf' # input("Introduce la ruta del PDF para el entrenamiento: ")
            train_model_from_pdf(pdf_path)
            model_output_dir = 'presets/' + pdf_path # input("Introduce la ruta para guardar los modelos entrenados: ")
            save_model_to_disk(model_output_dir)
            print("Modelos guardados exitosamente.")

        elif choice == "b":
            model_dir = 'presets/target.pdf' # input("Introduce la ruta del directorio con los modelos entrenados: ")
            model, tokenizer = load_model_from_disk(model_dir)

            while True:
                user_input = input("Usuario: ")
                input_ids = tokenizer.encode(user_input, return_tensors="pt")
                output = model.generate(input_ids, max_length=100, num_return_sequences=1)
                generated_response = tokenizer.decode(output[0], skip_special_tokens=True)

                # Split the input string by '\n\n' to create a list of paragraphs
                paragraphs = generated_response.split('.')

                # Get a random paragraph using random.choice()
                random_paragraph = random.choice(paragraphs)

                print("Modelo:", random_paragraph)

                continue_conversation = input("¿Quieres continuar conversando? (s/n): ")
                if continue_conversation.lower() != "s":
                    break

        elif choice == "c":
            break

        else:
            print("Opción inválida. Por favor, elige 'a', 'b' o 'c'.")

if __name__ == "__main__":
    main()
