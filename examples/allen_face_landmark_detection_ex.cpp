// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This example program shows how to find frontal human faces in an image and
    estimate their pose.  The pose takes the form of 68 landmarks.  These are
    points on the face such as the corners of the mouth, along the eyebrows, on
    the eyes, and so forth.  
    


    The face detector we use is made using the classic Histogram of Oriented
    Gradients (HOG) feature combined with a linear classifier, an image pyramid,
    and sliding window detection scheme.  The pose estimator was created by
    using dlib's implementation of the paper:
       One Millisecond Face Alignment with an Ensemble of Regression Trees by
       Vahid Kazemi and Josephine Sullivan, CVPR 2014
    and was trained on the iBUG 300-W face landmark dataset (see
    https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/):  
       C. Sagonas, E. Antonakos, G, Tzimiropoulos, S. Zafeiriou, M. Pantic. 
       300 faces In-the-wild challenge: Database and results. 
       Image and Vision Computing (IMAVIS), Special Issue on Facial Landmark Localisation "In-The-Wild". 2016.
    You can get the trained model file from:
    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2.
    Note that the license for the iBUG 300-W dataset excludes commercial use.
    So you should contact Imperial College London to find out if it's OK for
    you to use this model file in a commercial product.


    Also, note that you can train your own models using dlib's machine learning
    tools.  See train_shape_predictor_ex.cpp to see an example.

    


    Finally, note that the face detector is fastest when compiled with at least
    SSE2 instructions enabled.  So if you are using a PC with an Intel or AMD
    chip then you should enable at least SSE2 instructions.  If you are using
    cmake to compile this program you can enable them by using one of the
    following commands when you create the build project:
        cmake path_to_dlib_root/examples -DUSE_SSE2_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_SSE4_INSTRUCTIONS=ON
        cmake path_to_dlib_root/examples -DUSE_AVX_INSTRUCTIONS=ON
    This will set the appropriate compiler options for GCC, clang, Visual
    Studio, or the Intel compiler.  If you are using another compiler then you
    need to consult your compiler's manual to determine how to enable these
    instructions.  Note that AVX is the fastest but requires a CPU from at least
    2011.  SSE4 is the next fastest and is supported by most current machines.  
*/

/*
0. can detect face ? 
1. detect blink ->  dms? 
2. detect is front face ?
3. detect is yaw ?
4. write data to protobuf files
*/


#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>
#include <iostream>
#include <chrono>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "proto/allen_blink.pb.h"

using namespace dlib;
using namespace std;


// define two constants, one for the eye aspect ratio to indicate
// blink and then a second constant for the number of consecutive
// frames the eye must be below the threshold
const double EYE_AR_THRESH = 0.2;
const int EYE_AR_CONSEC_FRAMES = 10;

const double YAW_THRESH = 1.0;
const int YAW_FRAMES = 25;

const bool WRITE_DATA = false;
// ----------------------------------------------------------------------------------------

inline std::vector<image_window::overlay_line> allen_render_face_detections (
    const std::vector<full_object_detection>& dets,
    const rgb_pixel color = rgb_pixel(0,255,0)
)
{
    std::vector<image_window::overlay_line> lines;
    for (unsigned long i = 0; i < dets.size(); ++i)
    {
        DLIB_CASSERT(dets[i].num_parts() == 68 || dets[i].num_parts() == 5,
            "\t std::vector<image_window::overlay_line> render_face_detections()"
            << "\n\t You have to give either a 5 point or 68 point face landmarking output to this function. "
            << "\n\t dets["<<i<<"].num_parts():  " << dets[i].num_parts() 
        );

        const full_object_detection& d = dets[i];

        if (d.num_parts() == 5)
        {
            lines.push_back(image_window::overlay_line(d.part(0), d.part(1), color));
            lines.push_back(image_window::overlay_line(d.part(1), d.part(4), color));
            lines.push_back(image_window::overlay_line(d.part(4), d.part(3), color));
            lines.push_back(image_window::overlay_line(d.part(3), d.part(2), color));
        }
        else
        {
            // // Around Chin. Ear to Ear
            // for (unsigned long i = 1; i <= 16; ++i)
            //     lines.push_back(image_window::overlay_line(d.part(i), d.part(i-1), color));

            // Line on top of nose
            for (unsigned long i = 28; i <= 30; ++i)
                lines.push_back(image_window::overlay_line(d.part(i), d.part(i-1), color));

            // // left eyebrow
            // for (unsigned long i = 18; i <= 21; ++i)
            //     lines.push_back(image_window::overlay_line(d.part(i), d.part(i-1), color));
            // // Right eyebrow
            // for (unsigned long i = 23; i <= 26; ++i)
            //     lines.push_back(image_window::overlay_line(d.part(i), d.part(i-1), color));
            // // Bottom part of the nose
            // for (unsigned long i = 31; i <= 35; ++i)
            //     lines.push_back(image_window::overlay_line(d.part(i), d.part(i-1), color));
            // // Line from the nose to the bottom part above
            // lines.push_back(image_window::overlay_line(d.part(30), d.part(35), color));

            // Left eye
            for (unsigned long i = 37; i <= 41; ++i)
                lines.push_back(image_window::overlay_line(d.part(i), d.part(i-1), color));
            lines.push_back(image_window::overlay_line(d.part(36), d.part(41), color));

            // Right eye
            for (unsigned long i = 43; i <= 47; ++i)
                lines.push_back(image_window::overlay_line(d.part(i), d.part(i-1), color));
            lines.push_back(image_window::overlay_line(d.part(42), d.part(47), color));

            // Lips outer part
            for (unsigned long i = 49; i <= 59; ++i)
                lines.push_back(image_window::overlay_line(d.part(i), d.part(i-1), color));
            lines.push_back(image_window::overlay_line(d.part(48), d.part(59), color));

            // Lips inside part
            for (unsigned long i = 61; i <= 67; ++i)
                lines.push_back(image_window::overlay_line(d.part(i), d.part(i-1), color));
            lines.push_back(image_window::overlay_line(d.part(60), d.part(67), color));
        }
    }
    return lines;
}

double left_eye_aspect_ratio(const full_object_detection& fod)
{
    const point& pt36 = fod.part(36);
    const point& pt37 = fod.part(37);
    const point& pt38 = fod.part(38);
    const point& pt39 = fod.part(39);
    const point& pt40 = fod.part(40);
    const point& pt41 = fod.part(41);

    double a = (pt37.x() - pt41.x()) * (pt37.x() - pt41.x()) + (pt37.y() - pt41.y()) * (pt37.y() - pt41.y());
    a = sqrt(a);
    double b = (pt38.x() - pt40.x()) * (pt38.x() - pt40.x()) + (pt38.y() - pt40.y()) * (pt38.y() - pt40.y());
    b = sqrt(b);
    double c = (pt36.x() - pt39.x()) * (pt36.x() - pt39.x()) + (pt36.y() - pt39.y()) * (pt36.y() - pt39.y());
    c = sqrt(c);

    return (a + b) / (2.0 * c);
}

double right_eye_aspect_ratio(const full_object_detection& fod)
{
    const point& pt42 = fod.part(42);
    const point& pt43 = fod.part(43);
    const point& pt44 = fod.part(44);
    const point& pt45 = fod.part(45);
    const point& pt46 = fod.part(46);
    const point& pt47 = fod.part(47);

    double a = (pt43.x() - pt47.x()) * (pt43.x() - pt47.x()) + (pt43.y() - pt47.y()) * (pt43.y() - pt47.y());
    a = sqrt(a);
    double b = (pt44.x() - pt46.x()) * (pt44.x() - pt46.x()) + (pt44.y() - pt46.y()) * (pt44.y() - pt46.y());
    b = sqrt(b);
    double c = (pt42.x() - pt45.x()) * (pt42.x() - pt45.x()) + (pt42.y() - pt45.y()) * (pt42.y() - pt45.y());
    c = sqrt(c);

    return (a + b) / (2.0 * c);
}

bool std_front_face(const full_object_detection& fod)
{
    double nose_x = (fod.part(28).x() + fod.part(29).x() + fod.part(30).x()) / 3.0f;
    double left_eye_x = fod.part(36).x() + fod.part(37).x() + fod.part(38).x() + fod.part(39).x() + fod.part(40).x() + fod.part(41).x();
    left_eye_x /= 6.0;
    double right_eye_x = fod.part(42).x() + fod.part(43).x() + fod.part(44).x() + fod.part(45).x() + fod.part(46).x() + fod.part(47).x();
    right_eye_x /= 6.0;
    double eye_dist_x = std::fabs(left_eye_x - right_eye_x);
    double mid_eye_x = (left_eye_x + right_eye_x) / 2.0;
    double delta = std::fabs(mid_eye_x - nose_x);
    return delta < eye_dist_x * 0.25f;
}

double mouth_ear(const full_object_detection& fod)
{
    double len = (std::fabs(fod.part(53).x() - fod.part(49).x()) + std::fabs(fod.part(55).x() - fod.part(59).x()))  / 2.0;
    double h = 0;
    h += std::fabs(fod.part(49).y() - fod.part(59).y());
    h += std::fabs(fod.part(50).y() - fod.part(58).y());
    h += std::fabs(fod.part(51).y() - fod.part(57).y());
    h += std::fabs(fod.part(52).y() - fod.part(56).y());
    h += std::fabs(fod.part(53).y() - fod.part(55).y());
    h /= 5.0;
    return h / len;
}

// ----------------------------------------------------------------------------------------
// ./allen_face_landmark_detection_ex shape_predictor_68_face_landmarks.dat 1 ~/myproject/dms/BlinkDetect/14-MaleNoGlasses.avi
// ./allen_face_landmark_detection_ex shape_predictor_68_face_landmarks.dat 0 abc
int main(int argc, char** argv)
{
    try
    {
        // This example takes in a shape model file and then a list of images to
        // process.  We will take these filenames in as command line arguments.
        // Dlib comes with example images in the examples/faces folder so give
        // those as arguments to this program.
        if (argc == 1)
        {
            cout << "Call this program like this:" << endl;
            cout << "./face_landmark_detection_ex shape_predictor_68_face_landmarks.dat faces/*.jpg" << endl;
            cout << "\nYou can get the shape_predictor_68_face_landmarks.dat file from:\n";
            cout << "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
            return 0;
        }
        else if (argc < 4)
        {
            cout << "invalid parameters" << endl;
            return 0;
        }

        // We need a face detector.  We will use this to get bounding boxes for
        // each face in an image.
        frontal_face_detector detector = get_frontal_face_detector();
        // And we also need a shape_predictor.  This is the tool that will predict face
        // landmark positions given an image and face bounding box.  Here we are just
        // loading the model from the shape_predictor_68_face_landmarks.dat file you gave
        // as a command line argument.
        shape_predictor sp;
        deserialize(argv[1]) >> sp;

        cv::VideoCapture capture(0);
        if (std::atoi(argv[2]) == 1)
            capture.open(argv[3]);
        if (!capture.isOpened())
        {
            std::cout << "can not load the video" << std::endl;
            return -1;
        }

        auto start = std::chrono::system_clock::now();
        int frame_count = 0;

        std::vector<double> ear_vec;
        std::vector<int> blink_idx_vec;

        std::vector<double> mouth_ear_vec;
        std::vector<int> yaw_idx_vec;
        int COUNTER = 0;
        int TOTAL = 0;
        int COUNTER_YAW = 0;
        int TOTAL_YAW = 0;
        image_window win, win_faces;
        cv::Mat frame;
        while (capture.read(frame))
        {
            frame_count++;

            cv_image<bgr_pixel> img(frame);

            // cout << "processing image " << argv[i] << endl;
            // array2d<rgb_pixel> img;
            // load_image(img, argv[i]);
            // // Make the image larger so we can detect small faces.
            // pyramid_up(img);

            // // Now tell the face detector to give us a list of bounding boxes
            // // around all the faces in the image.
            // std::vector<rectangle> dets = detector(img);
            std::vector<rectangle> dets = detector(img);

            if (dets.empty())
            {
                std::cout << "Can't detect face!!!" << std::endl;
                continue;
            }

            // Now we will go ask the shape_predictor to tell us the pose of
            // each face we detected.
            std::vector<full_object_detection> shapes;
            for (unsigned long j = 0; j < dets.size(); ++j)
            {
                full_object_detection shape = sp(img, dets[j]); // 68 key points
                // You get the idea, you can get all the face part locations if
                // you want them.  Here we just store them in shapes so we can
                // put them on the screen.
                shapes.push_back(shape);

                if (!std_front_face(shape))
                {
                    std::cout << "NO Front face!" << std::endl;
                }

                double leaft_ear = left_eye_aspect_ratio(shape);
                double right_ear = right_eye_aspect_ratio(shape);
                double ear = (leaft_ear + right_ear) / 2.0f;
                double mouthear = mouth_ear(shape);
                std::cout << "YAW: " << mouthear << std::endl;
                mouth_ear_vec.emplace_back(mouthear);
                ear_vec.emplace_back(ear);

                if (ear < EYE_AR_THRESH)
                    COUNTER += 1;
                else
                {
                    if (COUNTER >= EYE_AR_CONSEC_FRAMES)
                    {
                        TOTAL += 1;
                        
                        blink_idx_vec.emplace_back(ear_vec.size() - 1);
                        std::cout << "*******************blink detected..." << std::endl;
                        std::cout << "*******************blink frames: " << COUNTER << std::endl;
                        COUNTER = 0;
                        std::cout << "blink num: " << TOTAL << std::endl;
                    }
                    else
                    {
                        COUNTER = 0;
                    }
                }

                if (mouthear > YAW_THRESH)
                    COUNTER_YAW += 1;
                else
                {
                    if (COUNTER_YAW >= YAW_FRAMES)
                    {
                        TOTAL_YAW += 1;
                        COUNTER_YAW = 0;
                        yaw_idx_vec.emplace_back(mouth_ear_vec.size() - 1);
                        // std::cout << "blink detected..." << std::endl;
                        // std::cout << "blink num: " << TOTAL << std::endl;
                         std::cout << "##############YAW detected! " << std::endl;
                    }
                }
            }

            // Now let's view our face poses on the screen.
            win.clear_overlay();
            win.set_image(img);
            win.add_overlay(dets, rgb_pixel(255,0,0));
            win.add_overlay(allen_render_face_detections(shapes));

            // We can also extract copies of each face that are cropped, rotated upright,
            // and scaled to a standard size as shown here:
            dlib::array<array2d<rgb_pixel> > face_chips;
            extract_image_chips(img, get_face_chip_details(shapes), face_chips);
            win_faces.set_image(tile_images(face_chips));

            auto end = std::chrono::system_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            double seconds = (duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
            if ( seconds >= 1.0f)
            {
                double fps = frame_count * 1.0f / seconds;
                std::cout << "FFPS: " << fps << std::endl;
                start = end;
                frame_count  = 0;
            }
        }

        if (WRITE_DATA)
        {
            std::ofstream ofs("blink.dat");
            if (ofs.is_open())
            {
                Blink blink;
                for (auto v : ear_vec)
                {
                    blink.add_ears(v);
                }

                for (auto v : blink_idx_vec)
                {
                    blink.add_blink_idx(v);
                }

                blink.set_blink_num(TOTAL);

                blink.SerializePartialToOstream(&ofs);

                ofs.close();
            }
            else
            {
                std::cout << "blink.dat open failed." << std::endl;
            }

            std::ofstream ofs_yaw("yaw.dat");
            if (ofs_yaw.is_open())
            {
                Yaw yaw;
                for (auto v : mouth_ear_vec)
                {
                    yaw.add_ears(v);
                }

                for (auto v : yaw_idx_vec)
                {
                    yaw.add_yaw_idx(v);
                }

                yaw.set_yaw_num(TOTAL_YAW);

                yaw.SerializePartialToOstream(&ofs_yaw);

                ofs_yaw.close();
            }
            else
            {
                std::cout << "yaw.dat open failed." << std::endl;
            }
        }
    }
    catch (exception& e)
    {
        cout << "\nexception thrown!" << endl;
        cout << e.what() << endl;
    }
}

// ----------------------------------------------------------------------------------------

