export const initState = {
  conversation: [],
  current: 0,
  chat: [
    {
      title: "Generate useLocalStorage",
      id: 321123123,
      ct: "2023-12-12",
      messages: [
        {
          content: "Hello, I'm ChatGPT! Ask me anything!",
          sentTime: "1682827639323",
          role: "user",
          id: 123,
        },
      ],
    },
    {
      title: "Testing",
      ct: "20-06-2024",
      id: 2381923,
      messages: [],
    },
  ],
  currentChat: 0,
  options: {
    account: {
      name: "ChatWithFalcon",
      avatar: "",
    },
    general: {
      language: "English",
      theme: "light",
      command: "COMMAND_ENTER",
      size: "normal",
    },
  },
  is: {
    typeing: false,
    config: false,
    fullScreen: true,
    sidebar: true,
    inputing: false,
    thinking: false,
    apps: true,
  },
  typeingMessage: {},
  version: "0.1.0",
  cotent: "",
};
