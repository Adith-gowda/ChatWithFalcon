import { useState, useEffect } from "react";
import { useGlobal } from "../context";

export function useMesssage() {
  const { currentChat, chat, is } = useGlobal();
  console.log(chat, currentChat, is.thinking)
  const [message, setMessage] = useState({ messages: [] });
  useEffect(() => {
    if (chat.length) {
      setMessage(chat[currentChat]);
    }
  }, [chat, is.thinking, currentChat]);
  return { message };
}
