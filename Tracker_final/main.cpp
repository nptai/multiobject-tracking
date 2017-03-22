#include "opencv\cv.h"
#include "opencv\highgui.h"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <vector>
using namespace cv;
using namespace std;
#define MINHESSIAN 400
#define WINDOW "Demo"
#define DEPTH 2
#define THRESHOLD 0.4
#define SCALE 0.1
#define DISTANCE 20


bool dragging = 0;
Point cursor;
RNG randome;
Mat pattern[DEPTH];
bool showAll = false;

struct Particle {
	float weight;
	Point pos;
	float scale;

	Particle(float _weight, Point& _pos, float _scale = 1)
		:weight(_weight), pos(_pos), scale(_scale) {}

	bool operator < (Particle* other) {
		return weight > other->weight;
	}
};

class Object {
	MatND hists[DEPTH];
	int nPart;
	int id;
	Vector<Particle*> parts;
	Size sizeReg;

	MatND getHistogramHSV(Mat &hsv) {
		MatND hist;
		int hbins = 32, sbins = 16, vbins = 32;
		int histSize[] = { hbins, sbins, vbins };
		float hranges[] = { 0, 180 };
		float sranges[] = { 0, 256 };
		float vranges[] = { 0, 256 };
		const float* ranges[] = { hranges, sranges, vranges };

		int channels[] = { 0, 1, 2 };
		calcHist(&hsv, 1, channels, Mat(), hist, 3, histSize, ranges);

		normalize(hist, hist, 1.0, 0, NORM_L1);
		return hist;
	}

	void initPart(Point& point) {

		for (int i = 0; i < nPart; ++i)
			parts.push_back(new Particle(1.0 / nPart, point));
	}

	float calcWeight(const Mat img, Point& point, Size size) {
		float alpha = 1;
		float weight = 0;
		for (int d = 0; d < DEPTH; ++d) {
			Rect roi(point - Point(size.width / 2, size.height / 2), size);
			size.width /= 2; size.height /= 2;

			MatND hist = getHistogramHSV(img(roi));
			weight += alpha*(1 - compareHist(hists[d], hist, CV_COMP_BHATTACHARYYA));
			alpha *= 4;
		}

		return 3*weight / (alpha - 1);
	}

	Particle* getBest() {
		Particle* best = parts[0];
		for (int i = 1; i < nPart; ++i)
			if (best->weight < parts[i]->weight)
				best = parts[i];

		return best;
	}



public:
	Object(int _id, const Rect &region, Mat &img, int _nPart) : nPart(_nPart), id(_id) {

		int w = region.width;
		int h = region.height;
		sizeReg = Size(w, h);
		Point g(region.x + w / 2, region.y + h / 2);

		for (int d = 0; d < DEPTH; ++d) {
			Rect roi(g - Point(w / 2, h / 2), Size(w, h));
			w /= 2; h /= 2;
			hists[d] = getHistogramHSV(img(roi)).clone();
		}

		initPart(g);
	}

	void update(Mat& img) {
		int height = img.rows;
		int width = img.cols;
		float W = 0;

		for (int i = 0; i < nPart; ++i) {
			Point newPoint = parts[i]->pos + Point(randome.gaussian(DISTANCE), randome.gaussian(DISTANCE));
			float newScale = abs(parts[i]->scale + randome.gaussian(SCALE));
			newScale = min(float(3.0), max(newScale, float(0.3)));

			int newHeight = sizeReg.height*newScale;
			int newWidth = sizeReg.width*newScale;

			// new particle outside img
			if (newHeight < 8 || newWidth < 8 || 
				newPoint.x - newWidth / 2 < 0 || newPoint.y - newHeight / 2 < 0 ||
				newPoint.x + newWidth / 2 >= width || newPoint.y + newHeight / 2 >= height) continue;

			parts[i]->pos = newPoint;
			parts[i]->scale = newScale;
			parts[i]->weight = calcWeight(img, newPoint, Size(newWidth, newHeight));
		}
	}

	void resample() {
		float* r = new float[nPart];
		r[0] = parts[0]->weight;
		for (int i = 1; i < nPart; ++i) {
			r[i] = r[i - 1] + parts[i - 1]->weight;
		}
		vector<float> p;
		for (int i = 0; i < nPart / 2; ++i) {
			p.push_back(randome.uniform(float(0), r[nPart - 1]));
		}

		sort(p.begin(), p.end());
		Particle** tmp = new Particle*[nPart / 2];
		int j = 0;
		for (int i = 0; i < nPart / 2; ++i) {
			while (r[j] < p[i]) j++;
			parts[i]->pos = parts[j]->pos;
			parts[i]->scale = parts[j]->scale;
			parts[i]->weight = 0;
		}

		j = 0;
		for (int i = nPart / 2; i < nPart; ++i) {
			j++;
			parts[i]->pos = parts[j]->pos;
			parts[i]->scale = parts[j]->scale;
			parts[i]->weight = 0;
		}


		/*sort(parts.begin(), parts.end());
		for (int i = nPart / 2; i < nPart; ++i)
		parts[i] = parts[i - nPart / 2];
		*/
	}

	void display(Mat& img) {
		Particle* best = getBest();
		printf("%.3f %d\n", best->weight, nPart);
		if (best->weight < THRESHOLD) {
			return;
		}
		Size newSize(sizeReg.width*best->scale, sizeReg.height*best->scale);
		Rect reg(best->pos - Point(newSize.width / 2, newSize.height / 2), newSize);
		rectangle(img, reg, CV_RGB(0, 0, 255));
		char* str = new char[3];
		itoa(id, str, 10);

		rectangle(img, reg.tl(), reg.tl() + Point(14, 14), CV_RGB(255, 255, 255), CV_FILLED);
		putText(img, str, reg.tl() + Point(0, 12), FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, CV_RGB(255, 0, 0), 1, 8);
		
		if (!showAll) return;

		for (int i = 0; i < nPart; ++i) {
			best = parts[i];
			Size newSize(sizeReg.width*best->scale, sizeReg.height*best->scale);
			rectangle(img, Rect(best->pos - Point(newSize.width / 2, newSize.height / 2), newSize), CV_RGB(0, 0, 255));
		}
	}

};

vector<Object*> objects;

void mouseHandler(int event, int x, int y, int flags, void* param) {
	if (event == CV_EVENT_LBUTTONDOWN && !dragging) {
		cursor = Point(x, y);
		dragging = 1;
	}

	if (event == CV_EVENT_MOUSEMOVE && dragging) {
		Mat img = ((Mat*)param)->clone();
		rectangle(img, cursor, Point(x, y), CV_RGB(0, 255, 0), 1, 8, 0);
		imshow(WINDOW, img);
	}

	if (event == CV_EVENT_LBUTTONUP && dragging) {
		Rect rect = Rect(cursor, Size(x - cursor.x, y - cursor.y));
		if (rect.height < 0 || rect.width < 0) return;
		// Lưu vào danh sách các object
		Mat img = ((Mat*)param)->clone();
		objects.push_back(new Object(objects.size() + 1, rect, *(Mat*)param, 100));

		imshow(WINDOW, img);
		dragging = 0;
	}

}

void showHelp() {
	printf("Help\n");
	printf("\t -Press 'p' to pause\n");
	printf("\t -Press 's' to select object\n");
	printf("\t -Press 'q' to quit\n");
	printf("\t -Press 'a' to show all particle\n");
}

int main(int argc, char *argv[]) {
	showHelp();
	namedWindow(WINDOW, CV_WINDOW_AUTOSIZE);

	
	VideoCapture cap;
	if (argc < 2)
		cap = VideoCapture(0);
	else
		cap = VideoCapture(argv[1]);

		int command = 0;
	while (command != 'q') {
		Mat frame;
		cap >> frame;
		Mat hsv;
		cvtColor(frame, hsv, COLOR_BGR2HSV);
		if (command == 's') {
			setMouseCallback(WINDOW, mouseHandler, &frame);
			while (waitKey() != 's');
		}

		for (int i = 0; i < objects.size(); ++i) {
			Object* object = objects[i];
			object->update(frame);
		}

		for (int i = 0; i < objects.size(); ++i) {
			Object* object = objects[i];
			object->display(frame);
			object->resample();
		}

		imshow(WINDOW, frame);
		command = waitKey(30);

		if ((char)command == 'p') {
			while (waitKey(10) != 'p');
		}

		if ((char)command == 'a') {
			showAll = 1 - showAll;
		}

	}
	return 0;
}