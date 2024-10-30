import openai

# Set your OpenAI API key
openai.api_key = ''

response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo-instruct",
  messages=[
    {
      "role": "system",
      "content": "Your expertise as a ServiceNow developer is paramount in assisting users seeking guidance on specific aspects of development. When a user inquires about a particular object type, identified by Business Rule, along with the relevant table named change_request, and specifies affected field names with their particular type  There's no value given by the user, please follow best servicenow practices., your task is to provide accurate, detailed responses adhering to ServiceNow development best practices.It 's crucial to ensure clarity in user queries. If a user's request lacks clarity, take the initiative to fully comprehend their needs before proposing a solution.Moreover, maintain relevance to the ServiceNow environment, ensuring responses align with the specified object type. In scenarios where users inquire about client script object types, which may not adhere to best ServiceNow development practices, it 's important to address this concern. For example, advising against using GlideRecord in client scripts due to potential performance issues, and suggesting alternatives like Script Includes or Business Rules. Always leverage the specified field names provided by the user to construct scripts, ensuring that responses are both tailored and efficient.By adhering to these best practices, you'll provide invaluable assistance in navigating ServiceNow development challenges."
    },
    {
      "role": "user",
      "content": "Act as an Expert ServiceNow Solution Architect your task is to think as an expert solution architect and provide sustainable and good solutions,\nContext: inMorphis is thinking of designing its own AI powered Chatbot initially for MVP, we think we should just build a simple application on inMorphis ServiceNow instance \"MSP Instance\" \nhere are the details of what we are thinking of this AI Chatbot which we named \"inMorphis Genie\" \n1. inMorphis Genie will be build on MSP Instance, on MSP instance we have all of the inMorphis UserIDs coming from Azure AD. However we want to consider two things,\na.only the developers that are allocated in projects should be able to see the genie,\nb.We want to expose it to end users - who will not have any licenses, meaning even if we are creating custom tables we dont want license cost associated.\nand we dont have Virtual agent on MSP so, we will not be using that either\n2. initally the plan around inMorphis Genie is that:\na. MSP instance will be integrated with openai.\nb. there will be a dropdown or something to choose the object, object being\na. Business rules,\nb. Client Script\nc. UI Action\nd. Script Include\nthere will be a string box where user can type in their prompt or what they require,their need and click on a button,\nthen we will take that prompt and send it to openAI, this is one time we are sending to openAI this is to refine the prompt, we will ask GPT to refine the prompt to get accurate results and no or very less hallucinations.\nthe context for this first time will be that openai should act as an prompt engineer and make this user request into a proper prompt.\nafter openAi then provides a prompt build from users request we will store that in a field \"refined prompt\" then we will send that refined prompt to openai to generate the script needed by the user.\na prompt according to us should have these things:\n1. Persona(this we will fix, as developer)\n2. Context(This we will also fix)\n3. Objective/Asks(this will a variable and will come from the string box where user will enter their requirement)\n4. output format(this will also be fixed)\n5. Validation points(this is also fixed and this is where we will do prompt engineering, we will set instructions here of best practices and results, for example do not use gr in scripts that could be one)\n\nafter that we will give user to thumbs down or up to get the feedback, if the response or script was useful or not, for thumbs down we will also provide a field to enter what was wrong, and we want to stores these to actually evalutate how much sucessfull inmorphis genie actually is \n\nthere is also one more thing, we want to know when a developer is asking for help, which project is he associated with, now we cannot have this automated because inmorphis is not storing this data anywhere, so suggest some options, we were thinking of creating a selection box same as when we were asking for object whether, Business rule, client script, ui action, script include, similarlry we can ask which project are you associated with, we can have dropdown and one others option for others we will give a string field to type.\nthis is what we were thinking if you think theres any better alternative or solution feel free to suggest but do not ask to map the projects to developers or to create additional tables\n\nconsidering all of this data now your task is to help Jatin come up with a MVP Data model, you will specify exactly the approach on how we will actually achieve this considering all of the challenges specified above.\nbe very accurate and specific and tell me how many custom tables in servicenow will be created which fields will be created, where will we refer them if we do,\nwhat kind of scripts,\nwhere will be store everything,\n\nthink as an expert serviceNow solution architect and comeup with a very detailed implementation Data model"
    }
  ],
  temperature=0,
  max_tokens=500,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)
def calculate_token_cost(num_tokens):
    # Assuming each token costs $0.005
    token_cost = num_tokens * 0.030
    return token_cost

num_tokens = 1000
cost = calculate_token_cost(num_tokens)
print(f"The cost for using {num_tokens} tokens is ${cost}")

print(response, cost)
