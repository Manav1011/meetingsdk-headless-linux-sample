#include "ZoomSDKAudioRawDataDelegate.h"
#include "../util/WebSocketClient.h"
#include <iostream>
#include "../util/json.hpp"
#include <vector>
#include <cstring>
#include <arpa/inet.h>  // for htonl
#include <openssl/evp.h> // for Base64 encoding
#include <openssl/evp.h>
#include <openssl/buffer.h>  // Include

extern WebSocketClient* g_webSocketClient;
using json = nlohmann::json;

std::string base64_encode(const uint8_t* buffer, size_t length) {
    BIO* bio, * b64;
    BUF_MEM* bufferPtr;
    b64 = BIO_new(BIO_f_base64());
    bio = BIO_new(BIO_s_mem());
    bio = BIO_push(b64, bio);
    BIO_set_flags(bio, BIO_FLAGS_BASE64_NO_NL); // No newline
    BIO_write(bio, buffer, length);
    BIO_flush(bio);
    BIO_get_mem_ptr(bio, &bufferPtr);
    std::string encodedData(bufferPtr->data, bufferPtr->length);
    BIO_free_all(bio);
    return encodedData;
}

ZoomSDKAudioRawDataDelegate::ZoomSDKAudioRawDataDelegate(bool useMixedAudio = false, bool transcribe = false) : m_useMixedAudio(useMixedAudio), m_transcribe(transcribe){
    server.start();
}

// void ZoomSDKAudioRawDataDelegate::onMixedAudioRawDataReceived(AudioRawData *data) {
//     if (!m_useMixedAudio) return;

//     // write to socket
//     if (m_transcribe) {
//         server.writeBuf(data->GetBuffer(), data->GetBufferLen());
//         return;
//     }

//     // or write to file
//     if (m_dir.empty())
//         return Log::error("Output Directory cannot be blank");


//     if (m_filename.empty())
//         m_filename = "test.pcm";


//     stringstream path;
//     path << m_dir << "/" << m_filename;

//     writeToFile(path.str(), data);
// }

void ZoomSDKAudioRawDataDelegate::onMixedAudioRawDataReceived(AudioRawData* data) {
    if (g_webSocketClient) {
        g_webSocketClient->sendBinary(data->GetBuffer(), data->GetBufferLen());
    }
}

void ZoomSDKAudioRawDataDelegate::onOneWayAudioRawDataReceived(AudioRawData* data, uint32_t node_id) {
    if (g_webSocketClient) {
        // Encode audio data in Base64
        std::string audioBase64 = base64_encode(reinterpret_cast<const uint8_t*>(data->GetBuffer()), data->GetBufferLen());

        // Create JSON object
        json audioPacket = {
            {"node_id", node_id},
            {"audio", audioBase64}
        };

        // Convert JSON to string
        std::string jsonString = audioPacket.dump();

        // Send JSON
        g_webSocketClient->send(jsonString);
    }
}

// void ZoomSDKAudioRawDataDelegate::onOneWayAudioRawDataReceived(AudioRawData* data, uint32_t node_id) {
//     if (m_useMixedAudio) return;

//     stringstream path;
//     path << m_dir << "/node-" << node_id << ".pcm";
//     writeToFile(path.str(), data);
// }

void ZoomSDKAudioRawDataDelegate::onShareAudioRawDataReceived(AudioRawData* data) {
    stringstream ss;
    ss << "Shared Audio Raw data: " << data->GetBufferLen() / 10 << "k at " << data->GetSampleRate() << "Hz";
    Log::info(ss.str());
}


void ZoomSDKAudioRawDataDelegate::writeToFile(const string &path, AudioRawData *data)
{
    static std::ofstream file;
	file.open(path, std::ios::out | std::ios::binary | std::ios::app);

	if (!file.is_open())
        return Log::error("failed to open audio file path: " + path);
	
    file.write(data->GetBuffer(), data->GetBufferLen());

    file.close();
	file.flush();

    stringstream ss;
    ss << "Writing " << data->GetBufferLen() << "b to " << path << " at " << data->GetSampleRate() << "Hz";

    Log::info(ss.str());
}

void ZoomSDKAudioRawDataDelegate::setDir(const string &dir)
{
    m_dir = dir;
}

void ZoomSDKAudioRawDataDelegate::setFilename(const string &filename)
{
    m_filename = filename;
}
