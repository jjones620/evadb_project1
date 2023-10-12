from inference import Model


model = Model()

while True:
    user_input = input("Enter a question or type 'exit' to quit: ")

    if user_input.lower() == 'exit':
        break

    response = model.ask_question(user_input)
    print(response)
