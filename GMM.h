#ifndef GMM_H_
#define GMM_H_
#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core\core.hpp>
class GMM
{
public:
	static const int K = 5;	//��˹ģ�͵�����
	GMM(cv::Mat& _model);	//GMM�Ĺ��캯������ model �ж�ȡ�������洢
	double possibility(int, const cv::Vec3d) const;	//����ĳ����ɫ����ĳ������Ŀ����ԣ���˹���ʣ�
	double tWeight(const cv::Vec3d) const;	//����������Ȩ��
	int choice(const cv::Vec3d) const;	//����һ����ɫӦ���������ĸ��������˹������ߵ��
	void learningBegin();	//ѧϰ֮ǰ�����ݽ��г�ʼ��
	void addSample(int, const cv::Vec3d);	//��ӵ����ĵ�
	void learningEnd();	//������ӵ����ݣ������µĲ������
private:
	void calcuInvAndDet(int);	//����Э���������������ʽ��ֵ
	cv::Mat model;	//�洢GMMģ��
	double *coefs, *mean, *cov;	//ÿ����˹�ֲ���Ȩ�ء���ֵ��Э����
	double covInv[K][3][3];	//Э�������
	double covDet[K];	//Э���������ʽ
	//����ѧϰ�����б����м����ݵı���
	double sums[K][3];	// sums[i][j]: ��i����˹�ɷ���jά������ɫֵ���ܺ�
	double prods[K][3][3];	// prods[i][p][q]: ��i����˹�ɷ���p��qά������ɫֵ�˻����ܺ�
	int sampleCounts[K];	// ÿ����˹�ɷֵ���������
	int totalSampleCount;	// ��������������
};
#endif
