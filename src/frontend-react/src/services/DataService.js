import { BASE_API_URL } from "./Common";
import axios from 'axios';
import { v4 as uuidv4 } from 'uuid';

console.log("BASE_API_URL:", BASE_API_URL)

// Create an axios instance with base configuration
const api = axios.create({
    baseURL: BASE_API_URL
});
// Add request interceptor to include session ID in headers
api.interceptors.request.use((config) => {
    var sessionId = localStorage.getItem('userSessionId');
    if (sessionId) {
        config.headers['X-Session-ID'] = sessionId;
    } else {
        sessionId = uuidv4();
        localStorage.setItem('userSessionId', sessionId);
        config.headers['X-Session-ID'] = sessionId;
    }
    return config;
}, (error) => {
    return Promise.reject(error);
});

const DataService = {
    Init: function () {
        // Any application initialization logic comes here
    },
    GetChats: async function (model, limit) {
        const url = limit != null ? "/" + model + "/chats?limit=" + limit : "/" + model + "/chats";
        return await api.get(url);
    },
    GetChat: async function (model, chat_id) {
        return await api.get("/" + model + "/chats/" + chat_id);
    },
    StartChatWithLLM: async function (model, message) {
        return await api.post("/" + model + "/chats", message);
    },
    ContinueChatWithLLM: async function (model, chat_id, message) {
        return await api.post("/" + model + "/chats/" + chat_id, message);
    },
    GetChatMessageImage: function (model, image_path) {
        return BASE_API_URL + "/" + model + "/" + image_path;
    },
}

export default DataService;