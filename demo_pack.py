if __name__ == '__main__':
    pass
    # # (1)
    # agent_demo = SimpleAgent()
    # agent_demo.model_init("deepseek-chat")
    # agent_demo.agent_init('forcast', [get_user_location, get_weather])
    # agent_demo.config_create("thread_id", "1")
    # user1 = r'what is the weather outside?'
    # user2 = r'thank you!'
    # agent_demo.context_set(user_id="1")
    # agent_demo.chat_once(user1)
    # agent_demo.chat_once(user2)

    # # (2)
    # agent = SimpleAgent()
    # agent.model_init(model_name='deepseek-chat', model_type='basic')
    # agent.model_init(model_name='deepseek-reasoner', model_type='advanced', timeout=120, max_tokens=4096)
    # agent.config_create("thread_id", "1")
    # '''
    # dynamic: Dynamic models are selected at runtime based on the current state and context.
    # This enables sophisticated routing logic and cost optimization.
    # '''
    # agent.agent_init(prompt_sys_name=FORCAST_PROMPT, tools_list=[get_user_location, get_weather],
    #                  # response_format=ResponseFormat_detailed,
    #                  middleware=['dynamic'])
    # user1 = r'what is the weather outside?'
    # user2 = r'Can you talk about the Solutions to the Tower of Hanoi problem?'
    # agent.context_set(user_id="1")
    # agent.chat_once(user1)
    # agent.chat_once(user2)

    # # (3)
    # agent = SimpleAgent()
    # agent.model_init(model_name='deepseek-chat', model_type='basic')
    # agent.config_create("thread_id", "1")
    # '''
    # user_role_prompt: The @dynamic_prompt decorator creates middleware that
    #                   generates system prompts dynamically based on the model request.
    # '''
    # agent.agent_init(tools_list=[get_user_location, get_weather],
    #                  middleware=['handle_tool_errors', 'user_role_prompt'])
    # user1 = r'Briefly explain machine learning.'
    # agent.context_set(user_id="1", user_role="expert")  # user_role="Beginner"
    # agent.chat_once(user1, response_type='stream')

    # # (4)
    # agent = SimpleAgent()
    # agent.model_init(model_name='deepseek-chat', model_type='basic')
    # agent.config_create("thread_id", "1")
    # '''
    # ToolStrategy uses artificial tool calling to generate structured output.
    # '''
    # agent.agent_init(tools_list=[search],
    #                  response_format=ToolStrategy(ContactInfo)) # ProviderStrategy(ContactInfo)
    # user1 = r'Extract contact info from: John Doe, john@example.com, (555) 123-4567'
    # agent.context_set(user_id="1", user_role="expert")
    # agent.chat_once(user1)

    # # (5)
    # agent = SimpleAgent()
    # agent.model_init(model_name='deepseek-chat', model_type='basic')
    # agent.config_create("thread_id", "1")
    # '''
    # Use middleware to define custom state when your custom state needs to be accessed
    # by specific middleware hooks and tools attached to said middleware.
    # '''
    # agent.agent_init(tools_list=[search],
    #                  middleware='CustomMiddleware')
    # user1 = r'I prefer technical explanations. What is MATH?'
    # user2 = r'What is an Apple?'
    # agent.context_set(user_id="1", user_role="expert")
    # '''
    # # The agent can now track additional state beyond messages
    # '''
    # agent.chat_once(user1, user_preferences={"style": "technical", "verbosity": "detailed"})
    # agent.chat_once(user2)
    # '''
    # -- CONSISTENT characteristic experience --
    # 用户: I prefer technical explanations
    # AI: 记录到 user_preferences: {style: technical}
    # 用户: Explain AI
    # AI: Artificial Intelligence involves neural networks with backpropagation algorithms... [使用技术性解释]
    # '''














