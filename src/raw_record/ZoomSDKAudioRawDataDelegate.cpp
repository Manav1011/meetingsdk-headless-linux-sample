#include "ZoomSDKAudioRawDataDelegate.h"
#include "../util/WebSocketClient.h"
#include "../util/UDPSocketClient.h"
#include <iostream>
#include "../util/json.hpp"
#include <vector>
#include <cstring>
#include <arpa/inet.h>  // for htonl
#include <openssl/evp.h> // for Base64 encoding
#include <openssl/evp.h>
#include <openssl/buffer.h>  // Include
#include <chrono>
#include <iomanip>
#include <sstream>
// #include "meeting_service_components/meeting_audio_interface.h"
// #include "meeting_service_components/meeting_participants_ctrl_interface.h"
#include "../Zoom.h" // Assuming Zoom.h contains the getMeetingService() method

#include "../Config.h"
#include <boost/asio.hpp>


extern WebSocketClient* g_webSocketClient;
extern UDPSocketClient* g_udpSocketClient;

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

std::string getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto timeT = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

    std::ostringstream oss;
    oss << std::put_time(std::gmtime(&timeT), "%Y-%m-%dT%H:%M:%S");
    oss << "." << std::setw(3) << std::setfill('0') << ms.count() << "Z";  // Milliseconds + UTC 'Z'
    
    return oss.str();
}


ZoomSDKAudioRawDataDelegate::ZoomSDKAudioRawDataDelegate(bool useMixedAudio = true, bool transcribe = false) : m_useMixedAudio(useMixedAudio), m_transcribe(transcribe){
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

// std::string ZoomSDKAudioRawDataDelegate::getMeetingID() {
//     IMeetingService* meetingService = Zoom::getInstance().getMeetingService();
//     if (meetingService) {
//         IMeetingInfo* meetingInfo = meetingService->GetMeetingInfo();
//         if (meetingInfo) {
//             const zchar_t* meetingID = meetingInfo->GetMeetingID();
//             if (meetingID) {
//                 return std::string(meetingID);
//             }
//         }
//     }
//     return "";
// }

void ZoomSDKAudioRawDataDelegate::onMixedAudioRawDataReceived(AudioRawData* data) {
    if (g_udpSocketClient) {
        std::string audioBase64 = base64_encode(reinterpret_cast<const uint8_t*>(data->GetBuffer()), data->GetBufferLen());
        std::string meetingID = Zoom::getInstance().getMeetingID();
        std::string timestamp = getCurrentTimestamp();
        nlohmann::json audioPacket = {
            {"action", "stream_mixed"},
            {"meeting_id", meetingID},
            {"audio", audioBase64},
            {"timestamp", timestamp}
        };
        std::string jsonString = audioPacket.dump();
        g_udpSocketClient->send(jsonString);
    }
    // if (g_webSocketClient) {
    //     std::string audioBase64 = base64_encode(reinterpret_cast<const uint8_t*>(data->GetBuffer()), data->GetBufferLen());
    //     // Get the meeting ID
    //     std::string meetingID = Zoom::getInstance().getMeetingID();
    //     std::string timestamp = getCurrentTimestamp();
    //     nlohmann::json audioPacket = {
    //         {"action","stream_mixed"},
    //         {"meeting_id", meetingID},
    //         {"audio", audioBase64},
    //         {"timestamp", timestamp}        
    //     };
    //     std::string jsonString = audioPacket.dump();
    //     g_webSocketClient->send(jsonString);
    // }
}

void ZoomSDKAudioRawDataDelegate::onOneWayAudioRawDataReceived(AudioRawData* data, uint32_t node_id) {
    if (g_udpSocketClient) {
        std::string audioBase64 = base64_encode(reinterpret_cast<const uint8_t*>(data->GetBuffer()), data->GetBufferLen());
        IMeetingParticipantsController* participantsController = Zoom::getInstance().getMeetingService()->GetMeetingParticipantsController();
        Config config;
        // Get the meeting ID
        std::string meetingID = Zoom::getInstance().getMeetingID();
        // std::cout << "Meeting ID: " << meetingID << std::endl;
        if (participantsController) {
            IUserInfo* userInfo = participantsController->GetUserByUserID(node_id);
            if (userInfo) {
                std::string userName = userInfo->GetUserName();
                std::string timestamp = getCurrentTimestamp();
                // std::cout << "User Name: " << userName << std::endl;
                if (userName != "ZoomBot"){
                    std::string timestamp = getCurrentTimestamp();
                    // Create JSON payload
                    nlohmann::json audioPacket = {
                        {"action","stream_individual"},
                        {"username", userName},
                        {"meeting_id", meetingID},
                        {"node_id", node_id},
                        {"audio", audioBase64},
                        {"timestamp", timestamp}  // Add timestamp
                    };
                    std::string jsonString = audioPacket.dump();
                    // Send JSON
                    g_udpSocketClient->send(jsonString);
                }
            }
        }
    }
    // if (g_webSocketClient) {
    //     // Encode audio data in Base64
    //     std::string audioBase64 = base64_encode(reinterpret_cast<const uint8_t*>(data->GetBuffer()), data->GetBufferLen());
    //     IMeetingParticipantsController* participantsController = Zoom::getInstance().getMeetingService()->GetMeetingParticipantsController();
    //     Config config;
    //     // Get the meeting ID
    //     std::string meetingID = Zoom::getInstance().getMeetingID();
    //     // std::cout << "Meeting ID: " << meetingID << std::endl;
    //     if (participantsController) {
    //         IUserInfo* userInfo = participantsController->GetUserByUserID(node_id);
    //         if (userInfo) {
    //             std::string userName = userInfo->GetUserName();
    //             std::string timestamp = getCurrentTimestamp();
    //             // std::cout << "User Name: " << userName << std::endl;
    //             if (userName != "ZoomBot"){
    //                 std::string timestamp = getCurrentTimestamp();
    //                 // Create JSON payload
    //                 nlohmann::json audioPacket = {
    //                     {"action","stream_individual"},
    //                     {"username", userName},
    //                     {"meeting_id", meetingID},
    //                     {"node_id", node_id},
    //                     {"audio", audioBase64},
    //                     {"timestamp", timestamp}  // Add timestamp
    //                 };
    //                 std::string jsonString = audioPacket.dump();
    //                 // Send JSON
    //                 g_webSocketClient->send(jsonString);
    //             }
    //         }
    //     }
    // }
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
