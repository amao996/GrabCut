#ifndef GMM_H_
#define GMM_H_
#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core\core.hpp>
class GMM
{
public:
	static const int K = 5;	//高斯模型的数量
	GMM(cv::Mat& _model);	//GMM的构造函数，从 model 中读取参数并存储
	double possibility(int, const cv::Vec3d) const;	//计算某个颜色属于某个组件的可能性（高斯概率）
	double tWeight(const cv::Vec3d) const;	//计算数据项权重
	int choice(const cv::Vec3d) const;	//计算一个颜色应该是属于哪个组件（高斯概率最高的项）
	void learningBegin();	//学习之前对数据进行初始化
	void addSample(int, const cv::Vec3d);	//添加单个的点
	void learningEnd();	//根据添加的数据，计算新的参数结果
private:
	void calcuInvAndDet(int);	//计算协方差矩阵的逆和行列式的值
	cv::Mat model;	//存储GMM模型
	double *coefs, *mean, *cov;	//每个高斯分布的权重、均值和协方差
	double covInv[K][3][3];	//协方差的逆
	double covDet[K];	//协方差的行列式
	//用于学习过程中保存中间数据的变量
	double sums[K][3];	// sums[i][j]: 第i个高斯成分在j维度上颜色值的总和
	double prods[K][3][3];	// prods[i][p][q]: 第i个高斯成分在p和q维度上颜色值乘积的总和
	int sampleCounts[K];	// 每个高斯成分的样本数量
	int totalSampleCount;	// 所有样本的总数
};
#endif
