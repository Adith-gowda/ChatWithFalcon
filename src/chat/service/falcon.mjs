import { createParser } from "eventsource-parser";
import { setAbortController } from "./abortController.mjs";

export const throwError = async (response) => {
  if (!response.ok) {
    let errorPayload = null;
    try {
      errorPayload = await response.json();
      console.log(errorPayload);
    } catch (e) {
      // ignore
    }
  }
};

export const fetchAction = async ({
  method = "POST",
  messages = [],
  options = {},
  signal,
}) => {
  console.log(messages[messages.length-1].content,);

  const ngrokURL = "https://71fd-34-145-78-108.ngrok-free.app";

  const data = {
    "inputs" : messages[messages.length-1].content,
    "parameters" : {
      "temperature": 0.1,
      "max_tokens": 200,
    }
  };

  try {
    const response = await fetch(`${ngrokURL}/generate/`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const responseData = await response.json();
    const generatedText = responseData.generated_text; // Adjust according to your actual response structure
    // const newans = generatedText.split("User:")[0];
    return generatedText;
  } catch (error) {
    console.error("Error fetching data:", error);
    // Handle error here
    return null;
  }

};

export const fetchStream = async ({
  options,
  messages,
  onMessage,
  onEnd,
  onError,
  onStar,
}) => {
  let answer = "";
  const { controller, signal } = setAbortController();
  console.log(signal, controller);
  const result = await fetchAction({ options, messages, signal }).catch(
    (error) => {
      onError && onError(error, controller);
    }
  );

  onMessage && onMessage(result, controller);

  await onEnd();
};
