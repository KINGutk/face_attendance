-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: May 01, 2026 at 05:32 PM
-- Server version: 10.4.32-MariaDB
-- PHP Version: 8.2.12

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `face_attendance_db`
--

-- --------------------------------------------------------

--
-- Table structure for table `admins`
--

CREATE TABLE `admins` (
  `id` int(11) NOT NULL,
  `username` varchar(100) NOT NULL,
  `password_hash` varchar(255) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `admins`
--

INSERT INTO `admins` (`id`, `username`, `password_hash`) VALUES
(1, 'admin', '240be518fabd2724ddb6f04eeb1da5967448d7e831c08c8fa822809f74c720a9');

-- --------------------------------------------------------

--
-- Table structure for table `attendance`
--

CREATE TABLE `attendance` (
  `id` int(11) NOT NULL,
  `student_id` int(11) DEFAULT NULL,
  `date` date DEFAULT NULL,
  `time` time DEFAULT NULL,
  `status` varchar(10) DEFAULT NULL,
  `class_id` int(11) DEFAULT NULL,
  `method` enum('face','qr','manual','mobile') DEFAULT 'face'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `attendance`
--

INSERT INTO `attendance` (`id`, `student_id`, `date`, `time`, `status`, `class_id`, `method`) VALUES
(193, 77, '2026-01-08', '08:51:41', 'Present', 62, ''),
(194, 73, '2026-01-08', '09:00:00', 'Absent', 62, ''),
(195, 77, '2026-01-08', '09:04:52', 'Present', 63, ''),
(196, 73, '2026-01-08', '09:14:00', 'Absent', 63, ''),
(197, 73, '2026-02-21', '08:50:00', 'Present', 62, 'manual'),
(198, 77, '2026-02-21', '08:50:00', 'Present', 62, 'manual');

-- --------------------------------------------------------

--
-- Table structure for table `classes`
--

CREATE TABLE `classes` (
  `id` int(11) NOT NULL,
  `subject_name` varchar(100) DEFAULT NULL,
  `day_of_week` varchar(10) DEFAULT NULL,
  `start_time` time DEFAULT NULL,
  `end_time` time DEFAULT NULL,
  `professor_id` int(11) DEFAULT NULL,
  `semester` varchar(50) NOT NULL DEFAULT '1st Semester'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `classes`
--

INSERT INTO `classes` (`id`, `subject_name`, `day_of_week`, `start_time`, `end_time`, `professor_id`, `semester`) VALUES
(62, 'statestic', 'Thursday', '08:50:00', '09:00:00', 8, '5th Semester'),
(63, 'bio', 'Thursday', '09:04:00', '09:40:00', 8, '5th Semester');

-- --------------------------------------------------------

--
-- Table structure for table `detection_logs`
--

CREATE TABLE `detection_logs` (
  `id` int(11) NOT NULL,
  `name` varchar(255) DEFAULT NULL,
  `roll_no` varchar(100) DEFAULT NULL,
  `subject` varchar(255) DEFAULT NULL,
  `status` varchar(50) DEFAULT NULL,
  `timestamp` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `leaves`
--

CREATE TABLE `leaves` (
  `id` int(11) NOT NULL,
  `student_id` int(11) NOT NULL,
  `start_date` date NOT NULL,
  `end_date` date NOT NULL,
  `application_text` text DEFAULT NULL,
  `status` enum('Pending','Approved','Rejected') DEFAULT 'Pending',
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  `subject_name` varchar(100) DEFAULT NULL,
  `application_purpose` varchar(50) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `leaves`
--

INSERT INTO `leaves` (`id`, `student_id`, `start_date`, `end_date`, `application_text`, `status`, `created_at`, `subject_name`, `application_purpose`) VALUES
(47, 77, '2026-02-18', '2026-02-19', 'im sick for today ', 'Pending', '2026-02-18 14:11:27', 'statestic', 'Sick');

-- --------------------------------------------------------

--
-- Table structure for table `professors`
--

CREATE TABLE `professors` (
  `id` int(11) NOT NULL,
  `name` varchar(100) DEFAULT NULL,
  `email` varchar(100) DEFAULT NULL,
  `password` varchar(255) DEFAULT NULL,
  `status` varchar(20) NOT NULL DEFAULT 'pending'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `professors`
--

INSERT INTO `professors` (`id`, `name`, `email`, `password`, `status`) VALUES
(8, 'prof yonus saib', 'ynunuskhattak@gmail.com', 'scrypt:32768:8:1$sHmu9SsAqm2nuNN3$8da66e8fe060d0a59e9ac9877643529f893ccb3f82e40180f69ead68c0a445d68d698089cb7e28e12bfb714a48f2d427d114c8a890431477b99af083e996cf69', 'approved'),
(9, 'prof irfan ', 'irfan@gmail.com', 'scrypt:32768:8:1$7uejoBs1stQ0uRxv$50b37596d515be6daee93b484a0f040c2786501d005d4726fdfe3949f7bb395e7fb0188119f94cc0a2af3dec1806a8a6f40fafdde0f548c2723417fb30540109', 'approved');

-- --------------------------------------------------------

--
-- Table structure for table `students`
--

CREATE TABLE `students` (
  `id` int(11) NOT NULL,
  `name` varchar(100) DEFAULT NULL,
  `roll_no` varchar(50) DEFAULT NULL,
  `semester` varchar(20) NOT NULL DEFAULT '1st Semester',
  `email` varchar(100) DEFAULT NULL,
  `image_path` varchar(255) DEFAULT NULL,
  `password` varchar(255) DEFAULT NULL,
  `status` enum('pending','approved','rejected') DEFAULT 'approved',
  `registration_date` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `students`
--

INSERT INTO `students` (`id`, `name`, `roll_no`, `semester`, `email`, `image_path`, `password`, `status`, `registration_date`) VALUES
(73, 'Naveed khan', '25476', '5th Semester', 'naveedkhan20242@gmail.com', 'C:\\face_attendance_web\\faces\\25476_Naveed_khan\\profile.jpg', 'scrypt:32768:8:1$gyVos5CptNZvAKsn$081605ec3a868376ae269219372d09168b5d2522cda25fd5b77125c226e4938ff41b6980b9a482a2af7c3eadaa70b8c47707e1cb919e06e0ec110fc2e0bbd8fc', 'approved', '2026-01-05 14:10:00'),
(74, 'zaman jan', '0987678', '2nd Semester', 'ynunuskhattak@gmail.com', 'C:\\face_attendance_web\\faces\\0987678_zaman_jan\\front.jpg', 'scrypt:32768:8:1$d5XkhwRJJlyh9hP3$e85977833e361c6526ae488b4343e8b330b8b2d5b74656c48e5ae04f7a4336a9477cfd4b12478a694c6d1dde40a4f71a7c1d6c88be7d5a060621c644e1df6fc0', 'approved', '2026-01-07 08:47:35'),
(75, 'prof yonus saib', '17201', '2nd Semester', 'profaftab@gmail.com', 'C:\\face_attendance_web\\faces\\17201_prof_yonus_saib\\front.jpg', 'scrypt:32768:8:1$d8HoAkYI7TUVJoqk$b43466c266f37a5b8fe1b1c3ddd120c030371ad632643b1a3125ab6c752ababd936e199d4fd5c7d9a690ab3fdb8f48efba73ff8a433d0cf0e1d1866a0bf64541', 'approved', '2026-01-07 09:06:53'),
(76, 'kaliwal', '123432', '2nd Semester', 'kaliwal@gmail.com', 'C:\\face_attendance_web\\faces\\123432_kaliwal\\front.jpg', 'scrypt:32768:8:1$khNCSVY31OP75fF7$a16f84296d657aaa260d475b65cb40a4d69270ed0da76b48bfb8471d16234716a66dafdccc79b46b2ea1ac00e7238be7a2d965d0846bc7c60bd6f9fc177b7734', 'approved', '2026-01-07 09:13:29'),
(77, 'Abdul kabeer ', '233706', '5th Semester', 'kabeer50005@gmail.com', 'C:\\face_attendance_web\\faces\\233706_Abdul_kabeer_\\front.jpg', 'scrypt:32768:8:1$pexyxfmZ6nSNb2aj$d0414aee3f9d3e71242d21ecadea84c0531ee2d4178d1ef24381ba87f163ace71d8855b502d57a2b52301d25d9b9ba895cb3a9c1562543feb454848def00b297', 'approved', '2026-01-07 10:05:24'),
(78, 'Kami', '3737473', '7th Semester', 'alam@gmail.com', 'C:\\face_attendance_web\\faces\\3737473_Kami\\front.jpg', 'scrypt:32768:8:1$syaNmYRgzrReTNkr$4c22777660a21025a1f4988fff9cbe437a3c5b7908c79210882cfd05eaf3b1f46b8556b4ec67029f8ccd048646992ffa2d7823c45888f7c10ab91d964d41e096', 'approved', '2026-01-08 06:18:07');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `admins`
--
ALTER TABLE `admins`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `username` (`username`);

--
-- Indexes for table `attendance`
--
ALTER TABLE `attendance`
  ADD PRIMARY KEY (`id`),
  ADD KEY `idx_attendance_class_date` (`class_id`,`date`),
  ADD KEY `idx_attendance_student_class_date` (`student_id`,`class_id`,`date`);

--
-- Indexes for table `classes`
--
ALTER TABLE `classes`
  ADD PRIMARY KEY (`id`),
  ADD KEY `fk_class_prof` (`professor_id`);

--
-- Indexes for table `detection_logs`
--
ALTER TABLE `detection_logs`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `leaves`
--
ALTER TABLE `leaves`
  ADD PRIMARY KEY (`id`),
  ADD KEY `idx_leaves_student_status` (`student_id`,`status`);

--
-- Indexes for table `professors`
--
ALTER TABLE `professors`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `students`
--
ALTER TABLE `students`
  ADD PRIMARY KEY (`id`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `admins`
--
ALTER TABLE `admins`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=2;

--
-- AUTO_INCREMENT for table `attendance`
--
ALTER TABLE `attendance`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=199;

--
-- AUTO_INCREMENT for table `classes`
--
ALTER TABLE `classes`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=64;

--
-- AUTO_INCREMENT for table `detection_logs`
--
ALTER TABLE `detection_logs`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `leaves`
--
ALTER TABLE `leaves`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=48;

--
-- AUTO_INCREMENT for table `professors`
--
ALTER TABLE `professors`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=10;

--
-- AUTO_INCREMENT for table `students`
--
ALTER TABLE `students`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=79;

--
-- Constraints for dumped tables
--

--
-- Constraints for table `attendance`
--
ALTER TABLE `attendance`
  ADD CONSTRAINT `attendance_ibfk_1` FOREIGN KEY (`student_id`) REFERENCES `students` (`id`) ON DELETE CASCADE ON UPDATE CASCADE,
  ADD CONSTRAINT `attendance_ibfk_2` FOREIGN KEY (`class_id`) REFERENCES `classes` (`id`) ON DELETE CASCADE;

--
-- Constraints for table `classes`
--
ALTER TABLE `classes`
  ADD CONSTRAINT `fk_class_prof` FOREIGN KEY (`professor_id`) REFERENCES `professors` (`id`);

--
-- Constraints for table `leaves`
--
ALTER TABLE `leaves`
  ADD CONSTRAINT `leaves_ibfk_1` FOREIGN KEY (`student_id`) REFERENCES `students` (`id`) ON DELETE CASCADE ON UPDATE CASCADE;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
