import { GoogleGenAI, Type, Schema } from "@google/genai";
import { Character, VocabularyItem } from "../types";

const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

const CHARACTER_DESCRIPTIONS: Record<Character, string> = {
  [Character.Chiikawa]: "Chiikawa (a small, white, round, cute creature with bear-like ears, blushing cheeks), kawaii vector art style",
  [Character.Hachiware]: "Hachiware (a small, white and blue cat-like creature with blue tips on ears), kawaii vector art style",
  [Character.Usagi]: "Usagi (a small, yellow rabbit-like creature with long ears, energetic expression), kawaii vector art style",
  [Character.Tanjiro]: "Tanjiro Kamado from Demon Slayer (super cute Q-version chibi style, big eyes, wearing green and black checkered haori, scar on forehead, kind smile)",
  [Character.Nezuko]: "Nezuko Kamado from Demon Slayer (super cute Q-version chibi style, big eyes, pink kimono, bamboo muzzle, long black hair with orange tips)",
  [Character.Zenitsu]: "Zenitsu Agatsuma from Demon Slayer (super cute Q-version chibi style, big eyes, yellow hair, yellow haori with white triangles, crying or sleeping cute face)",
  [Character.Inosuke]: "Inosuke Hashibira from Demon Slayer (super cute Q-version chibi style, big eyes, wearing a grey boar mask, shirtless, tiny muscular body but very kawaii)",
};

export const generateContent = async (
  character: Character,
  scenePrompt: string
): Promise<{ vocabulary: VocabularyItem[]; imageUrl: string | null }> => {
  
  // 1. Generate Image first
  const description = CHARACTER_DESCRIPTIONS[character];
  
  const imagePrompt = `A high quality, cute, vibrant illustration of ${description} in a ${scenePrompt} setting. 
  The composition should be 'wide shot' or 'rich in detail' to allow for at least 10 distinct objects to be identified.
  Style: Japanese kawaii aesthetic, soft shading, thick clean outlines, sticker art style, flat design.
  Ensure the main character is in the center but surrounded by interesting objects related to the scene.
  No text, speech bubbles, or labels in the image itself. Clean background.`;

  console.log("Generating image with prompt:", imagePrompt);

  const imageResponse = await ai.models.generateContent({
    model: 'gemini-2.5-flash-image',
    contents: {
      parts: [
        { text: imagePrompt }
      ]
    },
    config: {
      imageConfig: {
        aspectRatio: "1:1",
      }
    }
  });

  let imageUrl: string | null = null;
  let base64Image: string | null = null;
  let mimeType: string = 'image/png';

  if (imageResponse.candidates?.[0]?.content?.parts) {
    for (const part of imageResponse.candidates[0].content.parts) {
      if (part.inlineData && part.inlineData.data) {
        base64Image = part.inlineData.data;
        mimeType = part.inlineData.mimeType || 'image/png';
        imageUrl = `data:${mimeType};base64,${base64Image}`;
        break;
      }
    }
  }

  if (!imageUrl || !base64Image) {
    throw new Error("Failed to generate image.");
  }

  // 2. Analyze Image to get Vocabulary and Coordinates
  // We pass the generated image back to the model to detect objects.
  const analysisPrompt = `
    Analyze this image of ${character} in a ${scenePrompt} setting.
    Identify exactly 10 distinct items in the image for vocabulary learning.
    
    CRITICAL REQUIREMENTS:
    1. Total items: 10.
    2. At least 5 of these items must be labeled with 'Advanced' or 'IELTS' level English vocabulary words that describe the object or the concept represented by the object (e.g., instead of 'light', use 'Illumination' or 'Luminary' if appropriate, or describe a specific quality).
    3. The other 5 can be common nouns suitable for daily life.
    4. Do not overlap bounding boxes significantly.
    
    For each object, provide:
    1. 'english': The word in English.
    2. 'korean': The word in Korean (Hangul).
    3. 'chinese': The word in Chinese (Simplified).
    4. 'box2d': The bounding box of the object in the image (ymin, xmin, ymax, xmax) on a scale of 0 to 1000.
    
    Return a JSON array.
  `;

  const analysisSchema: Schema = {
    type: Type.ARRAY,
    items: {
      type: Type.OBJECT,
      properties: {
        english: { type: Type.STRING },
        korean: { type: Type.STRING },
        chinese: { type: Type.STRING },
        box2d: {
          type: Type.OBJECT,
          properties: {
            ymin: { type: Type.INTEGER },
            xmin: { type: Type.INTEGER },
            ymax: { type: Type.INTEGER },
            xmax: { type: Type.INTEGER },
          },
          required: ["ymin", "xmin", "ymax", "xmax"]
        }
      },
      required: ["english", "korean", "chinese", "box2d"],
    },
  };

  const vocabResponse = await ai.models.generateContent({
    model: 'gemini-2.5-flash',
    contents: {
      parts: [
        { inlineData: { mimeType: mimeType, data: base64Image } },
        { text: analysisPrompt }
      ]
    },
    config: {
      responseMimeType: "application/json",
      responseSchema: analysisSchema,
    },
  });

  const vocabulary: VocabularyItem[] = JSON.parse(vocabResponse.text || "[]");

  return {
    vocabulary,
    imageUrl
  };
};